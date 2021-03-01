import base64
import imaplib
import sys
import time
import html2text
import re
import yaml
import argparse
import quopri
import confluent_kafka as ck
import splunklib.client as client
import splunklib.results as results

SUBJECT_RGX = re.compile("(?<=\\nSubject:\\s)(.*?)(?=\\n)")
RECIPIENT_RGX = re.compile("(?<=\\nTo:\\s)(.*?)(?=\\n)")
EXT_SENDER_RGX = re.compile("(From: [\\S\\s]*?(?=To:))")
DATE_RGX = re.compile("(?<=\\nDate:\\s)(.*?)(?=\\n)")
EMAIL_RGX = re.compile("(?<=<)(.*)(?=>)")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

h = html2text.HTML2Text()
h.ignore_links = True

sub = re.compile("(?<=^Subject: )(.*)(?=\n)")
sub2 = re.compile("(?<=\\nSubject:\\s)(.*?)(?=\\n)")

recip = re.compile("(?<=^To: )(.*)(?=\n)")
recip2 = re.compile("(?<=\\nTo:\\s)(.*?)(?=\\n)")


sndr = re.compile("(?<=^FROM: )(.*)(?=\n)")
sndr2 = re.compile("(?<=\\nFrom:\\s)(.*?)(?=\\n)")

dt = re.compile("(?<=^Date: )(.*)(?=\n)")
dt2 = re.compile("(?<=\\nDate:\\s)(.*?)(?=\\n)")


class EmailContent:
    def __init__(self):
        self.sender = ""
        self.recipient = ""
        self.date = ""
        self.body = ""

    def get_delimited_row(self, sep="^"):
        pass


h = html2text.HTML2Text()
h.ignore_links = False


class SplunkAPIUtils:
    def __init__(self, splunk_conf_kwargs):
        self._service = client.connect(**splunk_conf_kwargs)

    def execute_query(self, search_query, kwargs):
        job = self._service.jobs.create(search_query, **kwargs)
        while True:
            while not job.is_ready():
                pass
            stats = {
                "isDone": job["isDone"],
                "doneProgress": float(job["doneProgress"]) * 100,
                "scanCount": int(job["scanCount"]),
                "eventCount": int(job["eventCount"]),
                "resultCount": int(job["resultCount"]),
            }

            status = (
                "\r Splunk search execution progress: %(doneProgress)03.1f%%   %(scanCount)d scanned   "
                "%(eventCount)d matched   %(resultCount)d results"
            ) % stats
            sys.stdout.write(status)
            sys.stdout.flush()
            if stats["isDone"] == "1":
                sys.stdout.write("\n\nDone!\n\n")
                break
            time.sleep(2)
        results = self._paginate_results(job)
        job.cancel()
        return results

    def _paginate_results(self, job):
        # paginate through a set of results
        resultCount = job["resultCount"]  # Number of results this job returned
        offset = 0
        # Start at result 0
        count = 1000
        result_l = []
        while offset < int(resultCount):
            kwargs_paginate = {"count": count, "offset": offset}
            # Get the search results
            blocksearch_results = job.results(**kwargs_paginate)
            for result in results.ResultsReader(blocksearch_results):
                if isinstance(result, dict):
                    # json_result = json.dumps(result)
                    result_l.append(result)
            # Increase the offset to get the next set of results
            offset += count
        return result_l


def kafka_sink(parsed_df, kafka_conf):
    producer = ck.Producer(kafka_conf["producer_conf"])
    json_str = parsed_df.to_json(orient="records", lines=True)
    json_recs = json_str.split("\n")
    for json_rec in json_recs:
        try:
            producer.poll(0)
            producer.produce(kafka_conf["publish_topic"], json_rec)
        except BufferError as be:
            producer.poll(0.1)
            print(be)
    producer.flush()


def extract_header(raw_email):
    sndr_val = sndr.findall(raw_email)
    if not sndr_val:
        sndr_val = sndr2.findall(raw_email)
    sender_email = sndr_val[0]
    if "<" in sender_email:
        sender_email = EMAIL_RGX.findall(sender_email)[0]
    recip_val = recip.findall(raw_email)
    if not recip_val:
        recip_val = recip2.findall(raw_email)
    recipient_email = recip_val[0]
    if "<" in recipient_email:
        recipient_email = EMAIL_RGX.findall(recipient_email)[0]
    dt_val = dt.findall(raw_email)
    if not dt_val:
        dt_val = dt2.findall(raw_email)
    date = dt_val[0]
    return (sender_email, recipient_email, date)


def html_as_text(raw_email):
    raw_email_tmp = raw_email.replace("=\r\n", "")
    raw_email_tmp = raw_email_tmp.replace("=\n", "")
    raw_email_tmp = raw_email_tmp.replace("=\r", "")
    html = re.search("<html(?s)(.*)</html>", raw_email_tmp, re.DOTALL).group(0)
    html_as_text = h.handle(html)
    return html_as_text


def extract_special_emails(raw_email):
    ec = EmailContent()
    has_base64_encode_type_two = False
    has_base64_encode_type_one = "Content-transfer-encoding: base64" in raw_email
    if not has_base64_encode_type_one:
        has_base64_encode_type_two = "Content-Transfer-Encoding: base64" in raw_email
        
    if (has_base64_encode_type_one or has_base64_encode_type_two) and "<html" in raw_email:
        try:
            if has_base64_encode_type_one:
                attachment = re.search(
                    "Content-transfer-encoding: base64(.*)", raw_email, re.DOTALL
                ).group(1)
            else:
                 attachment = re.search(
                    "Content-Transfer-Encoding: base64(.*)", raw_email, re.DOTALL
                ).group(1)
            attachment = attachment.split("\r\n--")[0]
            attachment = base64.b64decode(attachment).decode("utf-8")
            (sender_email, recipient_email, date) = extract_header(attachment)
            ec.sender = sender_email
            ec.recipient = recipient_email
            ec.date = date

            if "base64" in attachment:
                body = re.search(
                    "Content-Transfer-Encoding: base64(.*)", attachment, re.DOTALL
                ).group(1)
                body = body.split("\r\n--")[0]
                if "base64" in body:
                    body = re.search(
                        "Content-Transfer-Encoding: base64(.*)", body, re.DOTALL
                    ).group(1)
                    body = body.split("\r\n--")[0]
                body = base64.b64decode(body).decode("utf-8")
                if "<html" in body:
                    body = html_as_text(body)
                ec.body = body
            elif "<html" in attachment:
                body = html_as_text(body)
                ec.body = body
            else:
                ec.body = attachment
        except Exception:
            body = html_as_text(raw_email)
            (sender_email, recipient_email, date) = extract_header(raw_email)
            ec.sender = sender_email
            ec.recipient = recipient_email
            ec.date = date
            ec.body = body

    elif has_base64_encode_type_one or has_base64_encode_type_two:
        if has_base64_encode_type_one:
            attachment = re.search(
                "Content-transfer-encoding: base64(.*)", raw_email, re.DOTALL
            ).group(1)
        else:
            attachment = re.search(
                    "Content-Transfer-Encoding: base64(.*)", raw_email, re.DOTALL
                ).group(1)
        attachment = attachment.split("\r\n--")[0]
        try:
            attachment = base64.b64decode(attachment).decode("utf-8")
        except Exception:
            if "External email:" in raw_email:
                body = re.search("External email: (.*)", raw_email, re.DOTALL).group(0)
            else:
                body = re.search(
                    "X-Ms-Exchange-Processed-By-Bccfoldering: (.*)",
                    raw_email,
                    re.DOTALL,
                ).group(1)
            (sender_email, recipient_email, date) = extract_header(raw_email)
            ec.sender = sender_email
            ec.recipient = recipient_email
            ec.date = date
            ec.body = body
            return ec
        (sender_email, recipient_email, date) = extract_header(attachment)
        ec.sender = sender_email
        ec.recipient = recipient_email
        ec.date = date
        if "base64" in attachment:
            body = re.search(
                "Content-Transfer-Encoding: base64(.*)", attachment, re.DOTALL
            ).group(1)
            body = body.split("\r\n--")[0]
            if "base64" in body:
                body = re.search(
                    "Content-Transfer-Encoding: base64(.*)", body, re.DOTALL
                ).group(1)
                body = body.split("\r\n--")[0]
            body = base64.b64decode(body).decode("utf-8")
            if "<html" in body:
                body = html_as_text(raw_email)
            ec.body = body
    elif "Content-transfer-encoding: quoted-printable" in raw_email:
        body = html_as_text(raw_email)
        body = quopri.decodestring(body).decode("utf-8")
        (sender_email, recipient_email, date) = extract_header(raw_email)
        ec.sender = sender_email
        ec.recipient = recipient_email
        ec.date = date
        ec.body = body

    elif "<html" in raw_email:
        body = html_as_text(raw_email)
        (sender_email, recipient_email, date) = extract_header(raw_email)
        ec.sender = sender_email
        ec.recipient = recipient_email
        ec.date = date
        ec.body = body

    else:
        if "External email:" in raw_email:
            body = re.search("External email: (.*)", raw_email, re.DOTALL).group(0)
        else:
            body = re.search(
                "X-Ms-Exchange-Processed-By-Bccfoldering: (.*)", raw_email, re.DOTALL
            ).group(1)
        (sender_email, recipient_email, date) = extract_header(raw_email)
        ec.sender = sender_email
        ec.recipient = recipient_email
        ec.date = date
        ec.body = body
    return ec


def extract_email(message):
    ec = EmailContent()
    for part in message.walk():
        ctype = part.get_content_type()
        cdispo = str(part.get("Content-Disposition"))
        if ctype == "text/plain" and "attachment" not in cdispo:
            payload = part.get_payload(decode=True)
            try:
                message_str = base64.b64decode(payload).decode("utf-8")
                if message_str:
                    ec = extract_special_emails(message_str)
            except Exception:
                print("payload not in base64 encoded")
        if ctype == "text/html" and "attachment" not in cdispo:
            decoded_payload = str(part.get_payload(decode=True))
            body = h.handle(decoded_payload)  # html -> text
            message_str = message.as_string()
            ext_sender = EXT_SENDER_RGX.findall(message_str)
            recipient = RECIPIENT_RGX.findall(message_str)
            date = DATE_RGX.findall(message_str)
            if len(ext_sender) >= 1:
                if "<" in ext_sender[0]:
                    ext_sender = EMAIL_RGX.findall(ext_sender[0].replace("\\n", "\\s"))[
                        0
                    ]
                else:
                    ext_sender = ext_sender[0]
                ec.sender = ext_sender.split(">")[0]
            if len(recipient) >= 1:
                if "<" in recipient[0]:
                    ec.recipient = EMAIL_RGX.findall(
                        recipient[0].replace("\\n", "\\s")
                    )[0]
                else:
                    ec.recipient = recipient[0]
            if len(date) >= 1:
                ec.date = date[0]
            ec.body = body
    return ec


def connect_emailserver(user, pwd, server):
    """Connect to [the specified] mail server. Return an open connection"""
    conn = imaplib.IMAP4_SSL(server)
    try:
        conn.login(user, pwd)
    except imaplib.IMAP4.error:
        print("Failed to login")
        sys.exit(1)
    return conn


def send_email(gdf):
    pass


def load_yaml(yaml_file):
    """
    Returns a dictionary of a configuration contained in the given yaml file
    :param yaml_file: YAML configuration filepath
    :type yaml_file: str
    :return: config_dict: Configuration dictionary
    :rtype: dict
    """
    with open(yaml_file) as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict


def isvalid_conf(config, exec_mode):
    pass
    """
    if ('username' in config and config['username'].empty()) or 'username' not in config:
        raise Exception('Username is missing !!. Please provide username in the configuration.')
    if 'password' in config and config['password'].empty():
        continue
    else:
        raise Exception('Password is missing !!. Please provide password in the configuration.')
    if args.mode == "inference":
    """


def update_config_file(fname, config_dct):
    with open(fname, "w") as f:
        yaml.dump(config_dct, f)


def parse_arguments():
    """
    Parse script arguments
    """
    parser = argparse.ArgumentParser(description="Email phising detection process")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mode",
        choices=["inference", "training"],
        help="Execute process in inference/training mode",
        required=True,
    )
    required_args.add_argument(
        "-c",
        "--conf",
        help="Email server and model inference/ training configuration.",
        required=True,
    )
    args = parser.parse_args()
    return args

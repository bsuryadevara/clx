import os
import utils
import cudf
import time
import email
import traceback
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.header import decode_header
from clx.analytics.sequence_classifier import SequenceClassifier
from cuml.preprocessing.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)


class PhishingEmailDetection:
    def __init__(self, config):
        self.config = config
        self.seq_classifier = SequenceClassifier()

    def inference(self, body_series):
        pretrained_modelpath = self.config["model_conf"]["trained_model_dir"]
        self.seq_classifier.init_model(pretrained_modelpath)
        pred, prob = self.seq_classifier.predict(body_series)
        return pred, prob

    def training(self, gdf):
        splunk_conf = self.config["splunk_conf"]
        splunk_utils = utils.SplunkAPIUtils(splunk_conf["server_conf"])
        now = datetime.now()
        five_days_before = now - timedelta(4)
        now = datetime.strftime(now, utils.DATETIME_FORMAT)
        five_days_before = datetime.strftime(five_days_before, utils.DATETIME_FORMAT)
        kwargs = {"earliest_time": five_days_before, "latest_time": now}
        print(kwargs)
        query = splunk_conf["query"]
        results = splunk_utils.execute_query(query, kwargs)
        labels_gdf = cudf.DataFrame(results)

        labels_gdf = labels_gdf[labels_gdf["urgency"] != "low"]
        labels_gdf["sender"] = labels_gdf["sender"].str.lower()
        labels_gdf["reported_by"] = labels_gdf["reported_by"].str.lower()

        labels_gdf["urgency"] = labels_gdf["urgency"].str.replace("critical", "1")
        labels_gdf["urgency"] = labels_gdf["urgency"].str.replace("high", "1")
        labels_gdf["urgency"] = labels_gdf["urgency"].str.replace("informational", "0")
        labels_gdf["urgency"] = labels_gdf["urgency"].str.replace("medium", "0")
        labels_gdf["urgency"] = labels_gdf["urgency"].astype("int64")

        joined_df = labels_gdf.merge(
            gdf,
            how="left",
            right_on=["reportedby", "ext_sender"],
            left_on=["reported_by", "sender"],
        )
        stage1_joined_df = joined_df[~joined_df["reportedby"].isna()]
        print("stage1_joined_df: {}".format(stage1_joined_df.shape))

        unmatched_labels_df = joined_df[joined_df["reportedby"].isna()]
        del joined_df
        unmatched_labels_df = unmatched_labels_df[
            ["sender", "reported_by", "urgency", "subject"]
        ]
        unmatched_emails_df = labels_gdf.merge(
            gdf,
            how="right",
            right_on=["reportedby", "ext_sender"],
            left_on=["reported_by", "sender"],
        )
        unmatched_emails_df = unmatched_emails_df[
            unmatched_emails_df["reported_by"].isna()
        ]
        unmatched_emails_df = unmatched_emails_df[
            ["ext_sender", "reportedby", "subj", "date", "recipient", "body"]
        ]
        stage2_joined_df = unmatched_labels_df.merge(
            unmatched_emails_df,
            how="left",
            right_on=["reportedby", "subj"],
            left_on=["reported_by", "subject"],
        )

        stage2_joined_df = stage2_joined_df[~stage2_joined_df["reportedby"].isna()]

        result_df = cudf.concat([stage1_joined_df, stage2_joined_df])
        result_df = result_df.rename(columns={"urgency": "label", "body": "email"})
        result_df = result_df[["email", "label"]]

        del gdf
        del labels_gdf
        del unmatched_labels_df
        del stage2_joined_df

        seq_classifier = SequenceClassifier()
        print('model directory: {}'.format(self.config["model_conf"]["trained_model_dir"]))
        if self.config["model_conf"]["trained_model_dir"] is not None :
            seq_classifier.init_model(self.config["model_conf"]["trained_model_dir"])
        else:
            seq_classifier.init_model('bert-base-uncased')
        X_train, X_test, y_train, y_test = train_test_split(
            result_df, "label", train_size=0.8
        )
        seq_classifier.train_model(X_train["email"], y_train, epochs=15, batch_size=100)
        seq_classifier.evaluate_model(X_test["email"], y_test)
        print(result_df.head(10))
        print(result_df.shape)
        return seq_classifier


def extract_emails(messages, conn):
    gdf = cudf.DataFrame()
    parent_email_sndr_l = []
    email_uid_l = []
    date_l = []
    body_l = []
    subject_l = []
    recipient_l = []
    ext_sender_l = []
    count = 0
    for num in messages[0].split():
        # print(num)
        try:
            typ, data = conn.fetch(num, "(RFC822)")

            raw_email = data[0][1]
            email_message = email.message_from_bytes(raw_email, _class=EmailMessage)
            has_attachment = email_message.get("X-MS-Has-Attach")
            # print("has_attachment {}".format(has_attachment))
            processed_att = False
            subject = email_message.get("Subject")
            if "=?utf-8" in subject or "=?Windows-" in subject:
                decoded_header = decode_header(subject)[0]
                subject = decoded_header[0].decode(decoded_header[1])
            # handling iso seperately b/c sometime we get iso-8859-i in the subject
            if "=?iso-" in subject:
                subject = decode_header(subject)[0][0].decode("iso-8859-8")
            subject = subject.replace("\n", "").replace("\r", "").rstrip().lstrip()
            # print("subject: {}".format(subject))
            if not isinstance(raw_email, str):
                raw_email = raw_email.decode("utf-8")
            reported_by = utils.sndr.findall(raw_email)
            if not reported_by:
                reported_by = utils.sndr2.findall(raw_email)
            reported_by = reported_by[0]
            if "<" in reported_by:
                reported_by = utils.EMAIL_RGX.findall(reported_by)[0]
            # print("reported_by: {}".format(reported_by))
            # Iterate over only attachments
            try:
                for attachment in email_message.iter_attachments():
                    if attachment.is_multipart():  # verify attachment type is email
                        processed_att = True
                        email_content = utils.extract_email(attachment)
                        if email_content.body != "":
                            subject_l.append(subject)
                            ext_sender_l.append(email_content.sender)
                            recipient_l.append(email_content.recipient)
                            body_l.append(email_content.body)
                            date_l.append(email_content.date)
                            parent_email_sndr_l.append(reported_by)
                        else:
                            email_content = utils.extract_special_emails(raw_email)
                            subject_l.append(subject)
                            ext_sender_l.append(email_content.sender)
                            recipient_l.append(email_content.recipient)
                            body_l.append(email_content.body)
                            date_l.append(email_content.date)
                            parent_email_sndr_l.append(reported_by)
                        print(
                            "\n subject: {} \n ext_sender: {} \n recipient: {} \n date: {}".format(
                                subject,
                                email_content.sender,
                                email_content.recipient,
                                email_content.date,
                            )
                        )
            except Exception as e:
                traceback.print_exc()
                # print("Above exception is handled in the next steps")

            if has_attachment and not processed_att:
                email_content = utils.extract_special_emails(raw_email)
                subject_l.append(subject)
                ext_sender_l.append(email_content.sender)
                recipient_l.append(email_content.recipient)
                body_l.append(email_content.body)
                date_l.append(email_content.date)
                parent_email_sndr_l.append(reported_by)
                print(
                    "\n subject: {} \n ext_sender: {} \n recipient: {} \n date: {}".format(
                        subject,
                        email_content.sender,
                        email_content.recipient,
                        email_content.date,
                    )
                )
            email_uid_l.append(num)

        except Exception as e:
            count += 1
            print(raw_email)
            traceback.print_exc()
            print(
                "Error occured while processing attachment for email {}: {}".format(
                    email_message.get("From"), e
                )
            )
            continue

    gdf["subj"] = subject_l
    gdf["ext_sender"] = ext_sender_l
    gdf["recipient"] = recipient_l
    gdf["body"] = body_l
    gdf["date"] = date_l
    gdf["reportedby"] = parent_email_sndr_l
    if not gdf.empty:
        gdf["ext_sender"] = gdf["ext_sender"].str.lower()
        gdf["reportedby"] = gdf["reportedby"].str.lower()
        gdf["recipient"] = gdf["recipient"].str.lower()
        gdf["ext_sender"] = gdf["ext_sender"].replace("\n", "")
        gdf["ext_sender"] = gdf["ext_sender"].replace("\r", "")
    print(gdf.head(10))
    print("Number of emails successfully processed: {}".format(gdf.shape))
    print("Number of error emails: {}".format(count))

    return (gdf, email_uid_l)


def start_process(confpath, exec_mode):
    config = utils.load_yaml(confpath)
    try:
        utils.isvalid_conf(config, exec_mode)
    except Exception as e:
        raise Exception(e)
    # Connect to email server
    email_conf = config["email_conf"]
    conn = utils.connect_emailserver(
        email_conf["username"], email_conf["password"], email_conf["mailserver"]
    )
    print("Mailserver connection is successful !!")

    # verify if we have process specific folder from the email account
    if "folder" in email_conf:
        conn.select(email_conf["folder"])
    # Upper case search command
    search_cmd = "{}".format(email_conf["search"].upper())
    if exec_mode == "training":
        now = datetime.now()
        four_days_before = now - timedelta(4)
        since = datetime.strftime(four_days_before, "%d-%b-%Y")
        search_cmd = '(SINCE "{}")'.format(since)
    
    print('searching emails with command: {}'.format(search_cmd))

    (retcode, messages) = conn.search(None, search_cmd)
    if retcode == "OK":
        print('Started extracting email content...')
        (gdf, email_uid_l) = extract_emails(messages, conn)
        email_uid_l = []
        if not gdf.empty:
            ped = PhishingEmailDetection(config)
            if exec_mode == "inference":
                print('Executing inference part')
                pred, prob = ped.inference(gdf["body"])
                gdf["pred"] = pred.reset_index(drop=True)
                gdf["prob"] = prob.reset_index(drop=True)
                filename = "inference_{}.csv".format(time.strftime("%Y_%m_%d-%H-%M-%S"))
                output_filepath = os.path.join(
                    config["model_conf"]["inference_log_dir"], filename
                )
                gdf.to_csv(output_filepath, index=False)
                utils.kafka_sink(gdf, config["kafka_conf"])
            elif exec_mode == "training":
                print('Executing training part')
                model = ped.training(gdf)
                output_dir = "pytorch_model_{}.bin".format(
                    time.strftime("%Y_%m_%d_%H_%M_%S")
                )
                model_output_path = os.path.join(
                    config["model_conf"]["save_model_dir"], output_dir
                )
                model.save_model(model_output_path)
                config["model_conf"]["trained_model_dir"] = model_output_path
                utils.update_config_file(confpath, config)
            else:
                raise Exception("Invalid execution mode.")
        # mark them as read once the phishing detection training/inference process is successful.
        if email_conf["seen_flag"]:
            for id in email_uid_l:
                conn.store(id, "-FLAGS", "\\Seen")
    else:
        raise Exception("Email search query return code is not OK")
    conn.close()


if __name__ == "__main__":
    args = utils.parse_arguments()
    print("Arguments passed: {} ".format(args))
    start_process(args.conf, args.mode)



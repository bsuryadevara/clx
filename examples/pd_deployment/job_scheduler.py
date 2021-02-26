import schedule
import time
import phishing_email_detection
import traceback

def inference_job(confpath, exec_mode):
    print('****************************************')
    print('Started training at {}'.format(time.ctime()))
    print('****************************************')
    try:
        phishing_email_detection.start_process(confpath, exec_mode)
        print('Finished inference !!!')
    except Exception:
        traceback.print_exc()
        print('Finished inference with errors !!!')

def training_job(confpath, exec_mode):
    print('****************************************')
    print('Started training at {}'.format(time.ctime()))
    print('****************************************')
    try:
        phishing_email_detection.start_process(confpath, exec_mode)
        print('Finished training !!!')
    except Exception:
        traceback.print_exc()
        print('Finished training with errors !!!')

#schedule.every().hour.do(inference_job, confpath="conf.yaml", exec_mode="inference")
schedule.every().day.at("23:10").do(training_job, confpath='conf.yaml', exec_mode='training')


while True:
    schedule.run_pending()
    time.sleep(1)

import boto3
import os
import logging
from botocore.exceptions import ClientError

def upload_file(local_file,bucket,s3_file=None):
    '''Uploads a file to an S3 bucket'''

    if s3_file is None:
        s3_file = os.path.basename(local_file)

    #Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_file,bucket,s3_file)
        print(f"File {local_file} successfully uploaded to {bucket}/{s3_file}")
    except FileNotFoundError:
        print(f"File {local_file} not found")
    except Exception as e:
        print(f"Error uploading file: {e}")


if __name__=="__main__":
    local_file = "../../data/raw/creditcard.csv"
    bucket_name = "fraud-detection-mlops-snira"

    upload_file(local_file,bucket_name)



import json
import os
import boto3

s3 = boto3.client("s3")
s3Bucket = "seniorprojectbucketnoahl"

allowedExtensions = ["image/jpg", "image/jpeg", "image/png"]

def eventHandler(event, context):
    try:
        upload = json.loads(event['body'])
        filename = upload['file_name']
        filetype = upload['file_type']

        if filetype not in allowedExtensions:
            return {
                "statusCode": 400,
                "body": json.dumps(upload)
            }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
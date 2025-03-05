import json
import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
s3Bucket = "seniorprojectbucketnoahl"

allowedExtensions = ["image/jpg", "image/jpeg", "image/png"]

def eventHandler(event, context):
    try:
        body = event.get('body')
        if isinstance(body, str):
            upload = json.loads(body)
        elif isinstance(body, dict):
            upload = body
        else:
            raise Exception('Invalid event body format. JSON string or dictionary only.')
            
        filename = upload['file_name']
        filetype = upload['file_type']

        if filetype not in allowedExtensions:
            return {
                "statusCode": 400,
                "body": json.dumps(upload)
            }
        
        response= {
            'success': True,
            's3_url': presignedURL(s3Bucket, filename)
        }

        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
def presignedURL(bucketName, objectName, expiration=3600):
    try:
        return s3.generate_presigned_url('put_object', Params={'Bucket': bucketName, 'Key': objectName}, ExpiresIn=expiration)
    except ClientError as e:
        raise Exception(f'Error generating presigned URL: {e}')
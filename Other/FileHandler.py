import boto3
import tempfile
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def getS3Client():
  return boto3.client('s3', endpoint_url=os.getenv('R2_BUCKET_ENDPOINT'), 
                      aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'), 
                      aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'))

def createTempFile(fileBytes, extension):
  with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
    temp_file.write(fileBytes)
    temp_file_path = temp_file.name
  
  return temp_file_path

def get_public_url(bucket: str, key: str) -> str:

    endpoint = os.getenv('R2_BUCKET_ENDPOINT')
    return f"{endpoint}/{bucket}/{key}"

def uploadFile(filePath: str, storagePath: str, bucketName: str = "2n2d") -> str | None:
    s3Client = getS3Client()
    key = f"{storagePath}/{os.path.basename(filePath)}"
    try:
        try:
            s3Client.delete_object(Bucket=bucketName, Key=key)
        except s3Client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass  

        with open(filePath, 'rb') as file:
            s3Client.upload_fileobj(file, bucketName, key)

        return get_public_url(bucketName, key)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    
def getFileBinaryData(path: str, bucket: str) -> bytes | None:
    s3Client = getS3Client()
    try:
        presigned_url = s3Client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': path},
            ExpiresIn=3600  # seconds
        )
        response = requests.get(presigned_url)
        response.raise_for_status()
        print(response)
        return response.content
    except Exception as e:
        print(e)
        return None    
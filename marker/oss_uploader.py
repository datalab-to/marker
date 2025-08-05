
import os
import boto3
from botocore.config import Config
import string
import random
from datetime import datetime
from pathlib import Path
import traceback
from typing import Union
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class S3Client:
    def __init__(self, 
                 endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL"),
                 aws_access_key_id=os.getenv("AWS_S3_ACCESS_KEY_ID"),
                 aws_secret_access_key=os.getenv("AWS_S3_SECRET_ACCESS_KEY"),
                 region_name=os.getenv("AWS_S3_REGION_NAME"),
                 bucket=os.getenv("AWS_STORAGE_BUCKET_NAME")
        ):
        self.connection_data = {
            'endpoint_url': endpoint_url,
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': region_name,
        }
        self.bucket = bucket
        self.s3_session = boto3.client('s3', config=Config(s3={'addressing_style': 'virtual'}), **self.connection_data)
    
    @staticmethod
    def random_str():
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(random.choices(alphabet, k=6))
    
    def s3_upload_from_file(self, filename: str, filebytes: bytes):
        p = Path(filename)
        date = datetime.now().strftime('%Y%m%d')
        key = f'qilu-brain/wemol/{date}/{p.stem}_qilu-brain_{self.random_str()}{p.suffix}'
        return self._s3_upload(filebytes, key)
    
    def download_from_s3(self, key: str):
        try:
            response = self.s3_session.get_object(Bucket=self.bucket, Key=key)
            # Extract and read the file content from the streaming body
            content = response['Body'].read()
            return content
        except Exception as e:
            traceback.print_exc()
            print(f'download from s3 failed, cause {str(e)}')
            return None
    
    def delete_from_s3(self, key: str):
        try:
            self.s3_session.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            traceback.print_exc()
            print(f'delete from s3 failed, cause {str(e)}')
            return False
    
    def _s3_upload(self, body: Union[str, bytes], key: str):
        try:
            _ = self.s3_session.put_object(Body=body, Bucket=self.bucket, Key=key)
            # 生成url，不过期的
            url = self.s3_session.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=None,
                HttpMethod='GET'
            )
            return {
                'url': url,
                'key': key
            }
            
        except Exception as e:
            traceback.print_exc()
            print(f'upload object failed, cause {str(e)}')
            return {}

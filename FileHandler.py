from supabase import create_client, Client
from dotenv import load_dotenv
import os
import requests
import tempfile

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

async def getFileBinaryData(path:str, bucket:str):
  signed_url_response = supabase.storage.from_(bucket).create_signed_url(
    path=path,
    expires_in=3600  # seconds
  )

  download_url = signed_url_response.get("signedURL")
  response = requests.get(download_url)
  
  print(response)
  return response.content

async def uploadFile(filePath: str, storagePath: str, bucket: str = "rezult") -> str | None:
    """
    Uploads a file to Supabase storage, deleting the existing file if it exists.
    Returns the public URL on success, None on failure.
    """
    try:
        with open(filePath, "rb") as f:
            file_data = f.read()
        # Delete existing file (if it exists) - ignore errors if it doesn't exist
        try:
            supabase.storage.from_(bucket).remove([storagePath])
        except:
            pass
        response = supabase.storage.from_(bucket).upload(path=storagePath, file=file_data, file_options={"upsert": False})
        if response.error:
            return None

        public_url = supabase.storage.from_(bucket).get_public_url(storagePath)
        return public_url
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def createTempFile(fileBytes, extension):
  with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
    temp_file.write(fileBytes)
    temp_file_path = temp_file.name
  
  return temp_file_path
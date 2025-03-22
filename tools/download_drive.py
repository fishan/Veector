from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'your_credentials.json'  # Замените на ваш файл

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

drive_service = build('drive', 'v3', credentials=creds)

file_id = 'ВАШ_ID_ФАЙЛА'
request = drive_service.files().get_media(fileId=file_id)
file = io.BytesIO()
downloader = MediaIoBaseDownload(file, request)

done = False
while not done:
    status, done = downloader.next_chunk()
    print(f"Загрузка: {int(status.progress() * 100)}%")

with open("downloaded_file.ext", "wb") as f:
    f.write(file.getvalue())

print("Файл скачан!")
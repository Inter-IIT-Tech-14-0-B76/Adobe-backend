from dotenv import load_dotenv
import os

from pathlib import Path
import firebase_admin
from firebase_admin import credentials, auth


load_dotenv()

firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS")
cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred)

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
COMFY_WORKFLOW_DIR = Path(os.getenv("COMFY_WORKFLOW_DIR", "./workflows"))
DEFAULT_WORKSPACE = os.getenv("COMFY_DEFAULT_WORKSPACE", "default")


ENV = os.getenv("ENV", "dev").lower()

if ENV == "dev":
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./dev.db")
else:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost:5432/mydb",
    )


S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)


LOCAL_TMP_DIR = Path(os.getenv("CV_LOCAL_TMP_DIR", "/tmp/cv_images"))

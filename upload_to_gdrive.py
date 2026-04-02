"""
Upload the entire NN project folder to Google Drive.

Before running:
  1. pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
  2. Place credentials.json (from Google Cloud Console) in this folder
  3. Run:  python upload_to_gdrive.py

First run opens a browser for one-time Google sign-in.
Token is saved as token.json — subsequent runs skip the browser.
"""

import os
import sys
from pathlib import Path

# ── package check ─────────────────────────────────────────────────────────────
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("Missing packages. Run this first:\n")
    print("  pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client\n")
    sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
SCOPES          = ["https://www.googleapis.com/auth/drive.file"]
BASE_DIR        = Path(__file__).parent          # NN folder
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE      = BASE_DIR / "token.json"
GDRIVE_FOLDER   = "NN_Project"                   # name of folder created in Drive

# Files/folders to skip
SKIP_NAMES = {".git", "__pycache__", "token.json", "credentials.json",
              ".streamlit", "node_modules"}
SKIP_EXTS  = {".pyc", ".pyo"}

# ── auth ──────────────────────────────────────────────────────────────────────

def authenticate():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(f"\nERROR: credentials.json not found in:\n   {BASE_DIR}\n")
                print("Follow the setup steps in the script header then re-run.\n")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds

# ── drive helpers ─────────────────────────────────────────────────────────────

def create_folder(service, name, parent_id=None):
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    f = service.files().create(body=meta, fields="id").execute()
    return f["id"]


def upload_file(service, local_path: Path, parent_id: str):
    mime = "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    meta  = {"name": local_path.name, "parents": [parent_id]}
    service.files().create(body=meta, media_body=media, fields="id").execute()


def upload_folder(service, local_dir: Path, parent_id: str):
    """Recursively upload a local folder to Drive under parent_id."""
    for item in sorted(local_dir.iterdir()):
        if item.name in SKIP_NAMES or item.suffix in SKIP_EXTS:
            continue
        if item.is_dir():
            print(f"  [DIR]  {item.relative_to(BASE_DIR)}")
            sub_id = create_folder(service, item.name, parent_id)
            upload_folder(service, item, sub_id)
        else:
            print(f"  [FILE] {item.relative_to(BASE_DIR)}")
            upload_file(service, item, parent_id)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n-- NN Project -> Google Drive --------------------------")
    print("Authenticating...")
    creds   = authenticate()
    service = build("drive", "v3", credentials=creds)

    print(f"Creating Drive folder: '{GDRIVE_FOLDER}'...")
    root_id = create_folder(service, GDRIVE_FOLDER)

    print(f"Uploading files from:\n  {BASE_DIR}\n")
    upload_folder(service, BASE_DIR, root_id)

    print(f"\nUpload complete!")
    print(f"   Find your files in Google Drive -> '{GDRIVE_FOLDER}'")
    print(f"   Direct link: https://drive.google.com/drive/folders/{root_id}\n")


if __name__ == "__main__":
    main()

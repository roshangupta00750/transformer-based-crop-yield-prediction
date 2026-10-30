"""Download the driver CSVs exported by scripts/25_gee_era5land_export.py.

Reads the Google Drive folder the exports wrote to, using the persistent Earth
Engine credentials (they carry the Drive scope), and saves every CSV to
data/raw/drivers_v4/. Files already present with the same size are skipped.

Usage:
    python scripts/26_gee_drivers_download.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ee
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_FOLDER = "crop_yield_drivers_v4"
OUT = ROOT / "data/raw/drivers_v4"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=DRIVE_FOLDER)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--delete-after", action="store_true",
                    help="permanently delete each Drive file once its local copy's md5 matches")
    args = ap.parse_args()
    folder_name, out_dir = args.folder, Path(args.out)
    creds = ee.data.get_persistent_credentials()
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    folders = drive.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)").execute()["files"]
    if not folders:
        print(f"Drive folder {folder_name!r} not found (no export finished yet?)")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for folder in folders:
        page = None
        while True:
            res = drive.files().list(
                q=f"'{folder['id']}' in parents and trashed=false",
                fields="nextPageToken, files(id,name,size,md5Checksum)", pageToken=page).execute()
            names = [f["name"] for f in res["files"]]
            dups = sorted({n for n in names if names.count(n) > 1})
            if dups:
                sys.exit(f"duplicate file names on Drive, refusing to guess which is current: {dups}")
            for f in res["files"]:
                dest = out_dir / f["name"]
                if dest.exists() and dest.stat().st_size == int(f.get("size", -1)):
                    continue
                buf = io.FileIO(dest, "wb")
                dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=f["id"]))
                done = False
                while not done:
                    _, done = dl.next_chunk()
                buf.close()
                n_new += 1
                print(f"downloaded {f['name']} ({int(f.get('size', 0)) / 1e6:.1f} MB)")
                if args.delete_after:
                    import hashlib
                    if hashlib.md5(dest.read_bytes()).hexdigest() == f.get("md5Checksum"):
                        drive.files().delete(fileId=f["id"]).execute()
                    else:
                        print(f"  md5 mismatch, NOT deleted from Drive: {f['name']}")
            page = res.get("nextPageToken")
            if not page:
                break
    print(f"{n_new} new file(s) in {out_dir}")


if __name__ == "__main__":
    main()

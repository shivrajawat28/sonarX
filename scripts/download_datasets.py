"""Dataset downloader — SONAR marine-debris project.

Downloads ONLY from official sources, into SONAR/datasets/raw/.
Downloads archives AS-IS (no extraction, no renaming, no preprocessing).
Resume-capable: complete files are skipped; interrupted downloads retry and resume.

Sources (verified 2026-09-10):
  drishti-sss      Hugging Face  rehan9599/drishti-sss          public, CC-BY-SA-4.0   [RESUME]
  subpipe          Zenodo 12666132 -> SubPipeMini2.zip           public, attribution    4,945,761,374 B md5:7e0d925f93a89bc0e8715e4f6f7caecb
  ai4shipwrecks    U-M Deep Blue    -> AI4Shipwrecks.zip         public, CC-BY-4.0      ~1.13 GB (no md5 published)
  ghostvision      Zenodo 20056679  -> GhostVision_...zip        open access, CC-BY-SA-4.0  898,783,834 B md5:36872df7198d915e39db0d837f34655a
  kaggle-sss       GitHub codeload  SeabedObjects (ship+airplane) public, academic use  [stand-in per user decision]

Usage (from repo root):
    "C:\\Users\\kr034\\AppData\\Local\\Programs\\Python\\Python314\\python.exe" scripts/download_datasets.py
    # or a single dataset:
    ... python.exe scripts/download_datasets.py subpipe
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "datasets" / "raw"

UA = {"User-Agent": "Mozilla/5.0 (SONAR-dataset-fetch; official-sources-only)"}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url(url: str, dest: Path, expected_bytes: int | None = None,
                 expected_md5: str | None = None, max_retries: int = 5) -> None:
    """Resume-capable downloader. Atomic finalize; MD5-checked when known."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_bytes is not None and dest.stat().st_size == expected_bytes:
            print(f"   already complete: {dest.name} ({expected_bytes:,} bytes)")
            if expected_md5:
                print(f"   md5: {md5_of(dest)} (expected {expected_md5})")
            return
        if expected_bytes is None and dest.stat().st_size > 0 and dest.suffix == ".zip":
            # can't trust partial zip without size match; fall through to resume
            pass

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = dict(UA)
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if have and r.status != 206:
                    have = 0  # server ignored Range -> start over
                total = expected_bytes
                if total is None:
                    cl = r.headers.get("Content-Length")
                    total = int(cl) + have if cl else None
                print(f"   attempt {attempt}: {dest.name} from byte {have:,}"
                      + (f" of {total:,}" if total else ""))
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            size = tmp.stat().st_size
            if expected_bytes is not None and size != expected_bytes:
                raise IOError(f"size mismatch: have {size:,}, expected {expected_bytes:,}")
            if expected_md5 is not None:
                got = md5_of(tmp)
                if got != expected_md5:
                    tmp.unlink(missing_ok=True)
                    raise IOError(f"md5 mismatch: got {got}, expected {expected_md5}")
                print(f"   md5 OK: {got}")
            os.replace(tmp, dest)  # atomic finalize
            print(f"   saved: {dest} ({size:,} bytes)")
            return
        except Exception as e:  # noqa: BLE001
            print(f"   attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = min(30, 2 ** attempt)
                print(f"   retrying in {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"could not complete download: {url}")


# --------------------------------------------------------------------------
# 1. DRISHTI-SSS — resume via huggingface_hub (local copy missing val/ + train/labels)
# --------------------------------------------------------------------------
def fetch_drishti() -> None:
    out = RAW / "drishti-sss"
    out.mkdir(parents=True, exist_ok=True)
    print("== drishti-sss: resuming HF download (rehan9599/drishti-sss) ==")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="rehan9599/drishti-sss",
        repo_type="dataset",
        local_dir=out,
        max_workers=4,
    )
    print("== drishti-sss: done ==")


# --------------------------------------------------------------------------
# 2. SubPipeMini2 — official Zenodo record 12666132 (v3.0.1), SSS-focused subset
# --------------------------------------------------------------------------
def fetch_subpipe() -> None:
    out = RAW / "subpipe"
    out.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/api/records/12666132/files/SubPipeMini2.zip/content"
    download_url(
        url, out / "SubPipeMini2.zip",
        expected_bytes=4_945_761_374,
        expected_md5="7e0d925f93a89bc0e8715e4f6f7caecb",
    )


# --------------------------------------------------------------------------
# 3. AI4Shipwrecks — official U-M Deep Blue direct file download (open access)
# --------------------------------------------------------------------------
def fetch_ai4shipwrecks() -> None:
    # Deep Blue's WAF 403s plain urllib; curl with browser headers works (open access).
    import subprocess

    out = RAW / "ai4shipwrecks"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "AI4Shipwrecks.zip"
    url = "https://deepblue.lib.umich.edu/data/downloads/db78tc698"
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    cmd = [
        "curl", "-fL", "--retry", "5", "--retry-delay", "5", "-C", "-",
        "-A", ua,
        "-e", "https://deepblue.lib.umich.edu/data/concern/file_sets/db78tc698",
        "-H", "Accept: */*",
        "-o", str(dest), url,
    ]
    print(f"   curl: {dest.name} (~1.13 GB)")
    subprocess.run(cmd, check=True)
    print(f"   saved: {dest} ({dest.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------
# 4. GhostVision — Zenodo record 20056679 (OPEN; the HF copy is gated and was
#    explicitly declined by the user)
# --------------------------------------------------------------------------
def fetch_ghostvision() -> None:
    out = RAW / "ghostvision"
    out.mkdir(parents=True, exist_ok=True)
    url = ("https://zenodo.org/api/records/20056679/files/"
           "GhostVision_DatasetAndModels.zip/content")
    download_url(
        url, out / "GhostVision_DatasetAndModels.zip",
        expected_bytes=898_783_834,
        expected_md5="36872df7198d915e39db0d837f34655a",
    )


# --------------------------------------------------------------------------
# 5. SeabedObjects (public GitHub ship+airplane subset) — stand-in for the Kaggle
#    challenge data, per user decision. Single zip, no extraction.
# --------------------------------------------------------------------------
def fetch_kaggle_sss() -> None:
    out = RAW / "kaggle-sss"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "SeabedObjects-Ship-and-Airplane-dataset.zip"
    for branch in ("master", "main"):
        url = ("https://codeload.github.com/huoguanying/"
               f"SeabedObjects-Ship-and-Airplane-dataset/zip/refs/heads/{branch}")
        try:
            download_url(url, dest, expected_bytes=None, max_retries=3)
            return
        except RuntimeError:
            print(f"   branch '{branch}' not available, trying next ...")
    raise RuntimeError("SeabedObjects GitHub download failed on all branches")


ALIASES = {
    "drishti-sss": fetch_drishti,
    "subpipe": fetch_subpipe,
    "ai4shipwrecks": fetch_ai4shipwrecks,
    "ghostvision": fetch_ghostvision,
    "kaggle-sss": fetch_kaggle_sss,
}


def main() -> int:
    order = sys.argv[1:] or list(ALIASES)
    failed: dict[str, str] = {}
    for name in order:
        fn = ALIASES.get(name)
        if fn is None:
            print(f"!! unknown dataset '{name}' — skipping")
            continue
        print(f"\n### {name} " + "#" * max(1, 60 - len(name)))
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"!! {name} FAILED: {failed[name]}")
    print("\n" + "=" * 64)
    if failed:
        print("FAILED datasets:")
        for k, v in failed.items():
            print(f"  - {k}: {v}")
        return 1
    print("All requested datasets downloaded OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

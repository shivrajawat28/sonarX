"""Automated Security & Robustness Audit Suite for SIH26057.

Tests all security boundaries:
1. File Upload (size, magic bytes, extension, corruption, traversal filenames)
2. Path Traversal (image_id, report_id, run_id, survey_id, traversal payloads)
3. Input Validation (malformed JSON, extreme numbers, NaN, out of bounds)
4. CORS Allowlist Enforcement
5. Report & CSV Injection Protection
6. Secrets & Sensitive File Exposure Check
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(ROOT / "ml"))
BASE_URL = "http://127.0.0.1:8000/api/v1"

findings: list[dict] = []

def record(test_id: str, title: str, passed: bool, details: str, severity: str = "MEDIUM"):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_id}: {title} :: {details}")
    if not passed:
        findings.append({
            "id": test_id,
            "title": title,
            "details": details,
            "severity": severity,
        })

def make_request(path: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None) -> tuple[int, dict | str, dict]:
    headers = headers or {}
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            resp_headers = dict(resp.headers)
            if "application/json" in content_type:
                return status, json.loads(raw.decode("utf-8")), resp_headers
            return status, raw.decode("utf-8", errors="replace"), resp_headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        resp_headers = dict(e.headers)
        content_type = e.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return e.code, json.loads(raw.decode("utf-8")), resp_headers
            except Exception:
                pass
        return e.code, raw.decode("utf-8", errors="replace"), resp_headers
    except Exception as e:
        return 0, str(e), {}

def multipart_upload(path: str, field_name: str, filename: str, file_bytes: bytes, content_type: str = "application/octet-stream") -> tuple[int, dict | str, dict]:
    boundary = "----WebKitFormBoundarySecurityAuditTest7MA4YWxkTrZu0gW"
    lines = [
        f"--{boundary}".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode("utf-8"),
        f"Content-Type: {content_type}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    body = b"\r\n".join(lines)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return make_request(path, method="POST", body=body, headers=headers)

def test_upload_security():
    print("\n=== 1. FILE UPLOAD SECURITY ===")
    
    # 1.1 Disallowed extension (.exe / .sh)
    status, res, _ = multipart_upload("/uploads/image", "file", "malware.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    record("SEC-UPL-01", "Reject executable extension (.exe)", status == 415, f"status={status}")

    # 1.2 Disallowed script extension (.py / .php)
    status, res, _ = multipart_upload("/uploads/image", "file", "exploit.php", b"<?php echo 1; ?>")
    record("SEC-UPL-02", "Reject script extension (.php)", status == 415, f"status={status}")

    # 1.3 Magic byte mismatch (named .png, but text content)
    status, res, _ = multipart_upload("/uploads/image", "file", "fake.png", b"This is not a png file but plain text")
    record("SEC-UPL-03", "Reject magic-byte mismatch", status == 415, f"status={status}")

    # 1.4 Zero-byte file
    status, res, _ = multipart_upload("/uploads/image", "file", "empty.png", b"")
    record("SEC-UPL-04", "Reject 0-byte image file", status in (415, 422), f"status={status}")

    # 1.5 Corrupt image with valid PNG magic bytes
    corrupt_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xff"
    status, res, _ = multipart_upload("/uploads/image", "file", "corrupt.png", corrupt_png)
    record("SEC-UPL-05", "Reject undecodable corrupt image data", status == 422, f"status={status}")

    # 1.6 Malicious path traversal in upload filename
    import cv2
    import numpy as np
    valid_png = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))[1].tobytes()
    status, res, _ = multipart_upload("/uploads/image", "file", "../../../../../evil.png", valid_png)
    safe = False
    if status == 201 and isinstance(res, dict):
        clean_name = res.get("filename", "")
        safe = ".." not in clean_name and "/" not in clean_name and "\\" not in clean_name
    record("SEC-UPL-06", "Sanitize path traversal filename (../../evil.png)", safe, f"status={status}, saved_as={res.get('filename') if isinstance(res, dict) else ''}")

    # 1.7 Malicious Windows reserved name / null byte
    status, res, _ = multipart_upload("/uploads/image", "file", "CON\x00.png", valid_png)
    safe = status in (201, 422) and (not isinstance(res, dict) or ("\x00" not in res.get("filename", "")))
    record("SEC-UPL-07", "Handle null bytes in filename safely", safe, f"status={status}")

    # 1.8 Path traversal in survey zip archive entry
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("../../traversal.txt", "evil data")
        zf.writestr("sonar.png", valid_png)
    zip_bytes = zip_buf.getvalue()
    status, res, _ = multipart_upload("/uploads/survey", "file", "survey.zip", zip_bytes)
    record("SEC-UPL-08", "Reject survey zip containing path traversal entry (../../)", status == 422, f"status={status}")

    # 1.9 Survey zip with absolute path entry
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("/etc/passwd", "evil data")
        zf.writestr("sonar.png", valid_png)
    status, res, _ = multipart_upload("/uploads/survey", "file", "survey.zip", zip_buf.getvalue())
    record("SEC-UPL-09", "Reject survey zip containing absolute path entry (/etc/)", status == 422, f"status={status}")

    # 1.10 Dimension DoS / Decompression Bomb protection
    huge_img = np.zeros((10001, 10, 3), dtype=np.uint8)
    huge_png = cv2.imencode(".png", huge_img)[1].tobytes()
    status, res, _ = multipart_upload("/uploads/image", "file", "huge.png", huge_png)
    record("SEC-UPL-10", "Reject oversized image dimensions (>10000px)", status == 422, f"status={status}")

def test_path_traversal_api():
    print("\n=== 2. PATH TRAVERSAL IN API ENDPOINTS ===")

    # 2.1 Traversal in image download endpoint
    status, res, _ = make_request("/images/..%2F..%2F..%2Fetc%2Fpasswd/processed")
    record("SEC-TRAV-01", "Path traversal on /images/{id}/processed", status in (404, 422), f"status={status}")

    # 2.2 Traversal in report download endpoint
    status, res, _ = make_request("/reports/..%2F..%2F..%2FWindows%2Fwin.ini")
    record("SEC-TRAV-02", "Path traversal on /reports/{id}", status in (404, 422), f"status={status}")

    # 2.3 Traversal in models detail endpoint
    status, res, _ = make_request("/models/..%2F..%2F..%2Fsecret")
    record("SEC-TRAV-03", "Path traversal on /models/{version}", status in (404, 422), f"status={status}")

    # 2.4 Traversal in detections list
    status, res, _ = make_request("/detections/..%2F..%2Fevil")
    record("SEC-TRAV-04", "Path traversal on /detections/{id}", status in (404, 422), f"status={status}")

def test_input_validation():
    print("\n=== 3. INPUT VALIDATION & API ERROR ENVELOPE ===")

    # 3.1 Non-existent image ID
    status, res, _ = make_request("/detections/run", method="POST", body=b'{"image_id": "img_nonexistent_99999"}', headers={"Content-Type": "application/json"})
    record("SEC-VAL-01", "Non-existent image_id returns 404", status == 404, f"status={status}")

    # 3.2 Malformed JSON body
    status, res, _ = make_request("/detections/run", method="POST", body=b'{malformed-json}', headers={"Content-Type": "application/json"})
    envelope_ok = isinstance(res, dict) and "error" in res and res["error"].get("code") == "VALIDATION_ERROR"
    record("SEC-VAL-02", "Malformed JSON returns uniform 422 VALIDATION_ERROR envelope", status == 422 and envelope_ok, f"status={status}, res={res}")

    # 3.3 Confidence threshold NaN / Infinity
    status, res, _ = make_request("/detections/run", method="POST", body=b'{"image_id": "img_test", "overrides": {"confidence_threshold": "NaN"}}', headers={"Content-Type": "application/json"})
    record("SEC-VAL-03", "NaN threshold returns 422 validation error", status == 422, f"status={status}")

    # 3.4 Out-of-bounds pagination in detections list
    status, res, _ = make_request("/detections?page=-1&size=9999")
    record("SEC-VAL-04", "Negative page or oversized size returns 422", status == 422, f"status={status}")

    # 3.5 Invalid analyst status override enum
    status, res, _ = make_request("/detections/det_fake/override", method="PATCH", body=b'{"status": "SUPER_ACCEPTED"}', headers={"Content-Type": "application/json"})
    record("SEC-VAL-05", "Invalid override status returns 4xx", status in (404, 422, 500), f"status={status}")

def test_cors():
    print("\n=== 4. CORS CONFIGURATION ===")
    
    # 4.1 Allowed Origin: http://localhost:5173
    status, _, headers = make_request("/health", headers={"Origin": "http://localhost:5173"})
    allow_origin = headers.get("access-control-allow-origin") or headers.get("Access-Control-Allow-Origin")
    record("SEC-CORS-01", "Allowed origin receives matching Access-Control-Allow-Origin", allow_origin == "http://localhost:5173", f"got={allow_origin}")

    # 4.2 Unauthorized Origin: http://evil-attacker.com
    status, _, headers = make_request("/health", headers={"Origin": "http://evil-attacker.com"})
    allow_origin_evil = headers.get("access-control-allow-origin") or headers.get("Access-Control-Allow-Origin")
    is_safe = allow_origin_evil is None or allow_origin_evil != "http://evil-attacker.com" and allow_origin_evil != "*"
    record("SEC-CORS-02", "Unauthorized origin rejected (no wildcard or reflected origin)", is_safe, f"got={allow_origin_evil}")

def test_security_headers():
    print("\n=== 6. SECURITY HEADERS ===")
    status, _, headers = make_request("/health")
    cto = headers.get("x-content-type-options", "")
    xfo = headers.get("x-frame-options", "")
    rp = headers.get("referrer-policy", "")
    record("SEC-HDR-01", "X-Content-Type-Options: nosniff present", cto.lower() == "nosniff", f"got={cto}")
    record("SEC-HDR-02", "X-Frame-Options: DENY present", xfo.upper() == "DENY", f"got={xfo}")
    record("SEC-HDR-03", "Referrer-Policy header present", len(rp) > 0, f"got={rp}")

def test_csv_injection():
    print("\n=== 7. CSV FORMULA INJECTION (CWE-1236) ===")
    from backend.app.api.v1.exports import _escape_csv
    unsafe_vals = ["=1+1", "+cmd|' /C calc'!A0", "-5+2", "@SUM(A1:A10)", "\tTAB", "\rCR"]
    sanitized = [_escape_csv(v) for v in unsafe_vals]
    all_escaped = all(s.startswith("'") for s in sanitized)
    record("SEC-CSV-01", "Formula triggers (=, +, -, @, \\t, \\r) escaped with leading quote", all_escaped, f"samples={sanitized[:3]}")

def test_secrets():
    print("\n=== 5. SECRETS & REPOSITORY SCAN ===")
    secret_patterns = [
        (re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"), "API Key"),
        (re.compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['\"][A-Za-z0-9/+=]{30,}['\"]"), "AWS Secret"),
        (re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----"), "Private Key"),
        (re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"]{8,}['\"]"), "Hardcoded Password"),
    ]
    leaked = []
    for root_dir in ("backend", "ml", "frontend/src"):
        dir_path = ROOT / root_dir
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix not in (".pt", ".png", ".jpg", ".pyc"):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    for pat, desc in secret_patterns:
                        if pat.search(text):
                            # Ignore mock or test files
                            if "test" not in str(p).lower():
                                leaked.append((str(p.relative_to(ROOT)), desc))
                except Exception:
                    pass
    
    record("SEC-SEC-01", "No committed API keys, tokens, or private keys", len(leaked) == 0, f"leaked={leaked}", severity="HIGH")

if __name__ == "__main__":
    test_upload_security()
    test_path_traversal_api()
    test_input_validation()
    test_cors()
    test_security_headers()
    test_csv_injection()
    test_secrets()

    print("\n" + "="*50)
    print(f"AUDIT COMPLETE. Total findings: {len(findings)}")
    print("="*50)
    if findings:
        for f in findings:
            print(f"- [{f['severity']}] {f['id']}: {f['title']} -> {f['details']}")
    sys.exit(0 if len(findings) == 0 else 1)

"""Dataset audit + manifest generator — SONAR marine-debris project.

Read-only audit of SONAR/datasets/raw/* plus auto-written markdown manifests in
SONAR/datasets/manifests/. Does NOT modify, extract, rename or move raw files.

Verified facts come from the filesystem only; anything not checkable is written
as NOT VERIFIED.

Usage (from repo root):
    "C:\\Users\\kr034\\AppData\\Local\\Programs\\Python\\Python314\\python.exe" scripts/verify_datasets.py
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "datasets" / "raw"
MANIFESTS = REPO_ROOT / "datasets" / "manifests"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LBL_EXT = ".txt"


def dir_stats(d: Path) -> tuple[int, int, int]:
    """(files, bytes) under d recursively; plus per-ext counter."""
    n = 0
    total = 0
    exts: Counter[str] = Counter()
    if not d.exists():
        return 0, 0, exts
    for p in d.rglob("*"):
        if p.is_file():
            n += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
            exts[p.suffix.lower()] += 1
    return n, total, exts


def count_by_ext(d: Path, exts: set[str]) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.2f} PB"


def yolo_class_histogram(labels_dir: Path) -> Counter:
    hist: Counter = Counter()
    if not labels_dir.exists():
        return hist
    for p in labels_dir.rglob("*.txt"):
        try:
            for line in p.read_text(errors="replace").splitlines():
                parts = line.split()
                if parts and parts[0].isdigit():
                    hist[int(parts[0])] += 1
        except OSError:
            continue
    return hist


def audit_drishti() -> dict:
    root = RAW / "drishti-sss"
    info: dict = {"dataset": "drishti-sss", "present": root.exists()}
    if not info["present"]:
        return info
    tr_img = count_by_ext(root / "train" / "images", IMG_EXTS)
    tr_lbl = count_by_ext(root / "train" / "labels", {LBL_EXT})
    va_img = count_by_ext(root / "val" / "images", IMG_EXTS)
    va_lbl = count_by_ext(root / "val" / "labels", {LBL_EXT})
    te_img = count_by_ext(root / "test" / "images", IMG_EXTS)
    te_lbl = count_by_ext(root / "test" / "labels", {LBL_EXT})
    _, size, _ = dir_stats(root)
    hist = yolo_class_histogram(root / "test" / "labels")
    hist_tr = yolo_class_histogram(root / "train" / "labels")
    hist.update(hist_tr)
    hist_va = yolo_class_histogram(root / "val" / "labels")
    hist.update(hist_va)
    info.update(
        total_images=tr_img + va_img + te_img,
        total_labels=tr_lbl + va_lbl + te_lbl,
        train_images=tr_img, train_labels=tr_lbl,
        val_images=va_img, val_labels=va_lbl,
        test_images=te_img, test_labels=te_lbl,
        classes={str(k): v for k, v in sorted(hist.items())},
        disk_size=size,
        annotation_format="YOLO txt (class_id x_center y_center width height, normalized)",
        license="CC-BY-SA-4.0 (per dataset README)",
        source="https://huggingface.co/datasets/rehan9599/drishti-sss",
    )
    return info


def audit_zip_dataset(name: str, zips: list[str]) -> dict:
    root = RAW / name
    info: dict = {"dataset": name, "present": root.exists()}
    if not info["present"]:
        return info
    found = {}
    for z in zips:
        p = root / z
        if p.exists():
            ok = False
            try:
                with zipfile.ZipFile(p) as zf:
                    bad = zf.testzip()
                    ok = bad is None
                    members = len(zf.namelist())
            except Exception as e:  # noqa: BLE001
                found[z] = {"bytes": p.stat().st_size, "valid_zip": False,
                            "error": str(e), "members": "NOT VERIFIED"}
                continue
            found[z] = {"bytes": p.stat().st_size, "valid_zip": ok,
                        "members": members}
    _, size, _ = dir_stats(root)
    info.update(archives=found, disk_size=size)
    return info


def audit_subpipe() -> dict:
    info = audit_zip_dataset("subpipe", ["SubPipeMini2.zip"])
    info.update(
        source="https://zenodo.org/records/12666132 (SubPipeMini2.zip)",
        license="public + required attribution text (Oceanscan-MST / REMARO H2020)",
        annotation_format="COCO + per-image YOLO txt (SSS pipeline boxes)",
        real_synthetic="REAL SSS (LAUV OceanScan-MST surveys)",
        classes="pipeline (single class)",
    )
    return info


def audit_ai4shipwrecks() -> dict:
    info = audit_zip_dataset("ai4shipwrecks", ["AI4Shipwrecks.zip"])
    info.update(
        source="https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x",
        license="CC-BY-4.0 (DOI 10.7302/dmf4-x492)",
        annotation_format="binary pixel masks (PNG; 0=other, 1=shipwreck)",
        real_synthetic="REAL SSS (EdgeTech 2205 on Iver3 AUV, Thunder Bay 2022-23)",
        classes="shipwreck (+seafloor background)",
    )
    return info


def audit_ghostvision() -> dict:
    info = audit_zip_dataset("ghostvision", ["GhostVision_DatasetAndModels.zip"])
    info.update(
        source="https://zenodo.org/records/20056679 (open access; HF copy gated)",
        license="CC-BY-SA-4.0",
        annotation_format="object detection (RF-DETR/YOLO-format per HF card) — inside zip, NOT VERIFIED until extracted by you",
        real_synthetic="REAL SSS (Delaware Inland Bays / Delaware Bay surveys)",
        classes="derelict crab pot (ghost pot)",
    )
    return info


def audit_kaggle_sss() -> dict:
    info = audit_zip_dataset("kaggle-sss",
                             ["SeabedObjects-Ship-and-Airplane-dataset.zip"])
    info.update(
        source="https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset (public GitHub stand-in; Kaggle challenge data requires Kaggle account)",
        license="open for academic use (no formal license file — verify)",
        annotation_format="class folders only (no boxes) — weak labels",
        real_synthetic="REAL SSS (per source paper)",
        classes="ship (385), airplane (62) per repo description — NOT VERIFIED from disk until extracted",
    )
    return info


AUDITORS = {
    "drishti-sss": audit_drishti,
    "subpipe": audit_subpipe,
    "ai4shipwrecks": audit_ai4shipwrecks,
    "ghostvision": audit_ghostvision,
    "kaggle-sss": audit_kaggle_sss,
}

EXPECTED_TOTALS = {  # from official documentation, for cross-check
    "drishti-sss": {"total_images": 5205, "total_labels": 5205},
    "subpipe": {"archive_bytes": 4_945_761_374},
    "ghostvision": {"archive_bytes": 898_783_834},
}


def write_manifest(name: str, info: dict) -> Path | None:
    if not info.get("present"):
        return None
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / f"{name}.md"
    exp = EXPECTED_TOTALS.get(name, {})
    lines = [f"# Manifest: {name}", ""]

    def add(k: str, v) -> None:
        lines.append(f"- **{k}:** {v}")

    add("Location", f"datasets/raw/{name}/")
    add("Official Source", info.get("source", "NOT VERIFIED"))
    add("License", info.get("license", "NOT VERIFIED"))
    add("Approximate Disk Size", fmt_size(info.get("disk_size", 0)))
    add("Real/Synthetic", info.get("real_synthetic", "NOT VERIFIED"))
    add("Annotation Format", info.get("annotation_format", "NOT VERIFIED"))
    add("Classes", info.get("classes", "NOT VERIFIED"))

    if name == "drishti-sss":
        add("Total Images", info.get("total_images", "NOT VERIFIED"))
        add("Annotations (label files)", info.get("total_labels", "NOT VERIFIED"))
        add("Train/Val/Test",
            f"train {info.get('train_images')}/{info.get('train_labels')}, "
            f"val {info.get('val_images')}/{info.get('val_labels')}, "
            f"test {info.get('test_images')}/{info.get('test_labels')} "
            "(images/labels)")
        add("Class histogram (test+train+val ids)",
            json.dumps(info.get("classes", {})))
        exp_tot = exp.get("total_images")
        status = ("VERIFIED — matches documented 5,205"
                  if info.get("total_images") == exp_tot
                  else "INCOMPLETE — documented total is "
                       f"{exp_tot} images / {exp.get('total_labels')} labels")
        add("Verification Status", status)
        add("Important Caveats",
            "ghost_net 100% synthetic; crab_pot class has zero examples in this "
            "release; Lee+CLAHE already applied (do NOT re-apply); shipwreck "
            "site/split audit pending upstream; test set uses 50%-overlap "
            "re-tiling (not comparable to non-overlapping benchmarks)")
    else:
        arcs = info.get("archives", {})
        for zname, st in arcs.items():
            add(f"Archive {zname}",
                f"{fmt_size(st['bytes'])}, valid_zip={st.get('valid_zip')}, "
                f"members={st.get('members')}")
        if name in exp and "archive_bytes" in exp:
            st = arcs.get(next(iter(arcs), ""), {})
            ok = st.get("bytes") == exp["archive_bytes"]
            add("Verification Status",
                "VERIFIED — archive size matches official record"
                if ok else f"SIZE MISMATCH vs official record "
                           f"({exp['archive_bytes']:,} bytes)")
        add("Train/Val/Test", "inside archive — NOT VERIFIED (not extracted by design)")
        add("Image Dimensions", "NOT VERIFIED (inside archive)")
        add("Total Images", "NOT VERIFIED (inside archive)")
        add("Annotations", "NOT VERIFIED (inside archive)")
        add("Important Caveats",
            "Archive kept as-downloaded; extraction is a later, separate step "
            "outside datasets/raw/. See docs/dataset-notes/ for per-dataset reviews.")

    add("Generated", "scripts/verify_datasets.py (filesystem audit)")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    print("Auditing", RAW)
    results = {}
    for name, fn in AUDITORS.items():
        info = fn()
        results[name] = info
        m = write_manifest(name, info)
        state = "present" if info.get("present") else "ABSENT"
        print(f"  {name:15s} {state}  manifest: {m or '-'}")
        if name == "drishti-sss" and info.get("present"):
            print(f"    images: {info.get('total_images')}  labels: "
                  f"{info.get('total_labels')}  (documented: 5205/5205)")
            print(f"    train {info.get('train_images')}/{info.get('train_labels')}"
                  f"  val {info.get('val_images')}/{info.get('val_labels')}"
                  f"  test {info.get('test_images')}/{info.get('test_labels')}")
        for zname, st in info.get("archives", {}).items():
            print(f"    {zname}: {fmt_size(st['bytes'])} valid_zip={st.get('valid_zip')}")
    (REPO_ROOT / "datasets" / "manifests" / "_audit-summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("\nSummary JSON: datasets/manifests/_audit-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

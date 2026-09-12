"""Dataset manifests: reproducible dataset identity (Section 6.2).

A manifest records every image/label file with its content hash, the resolved
class list, and distribution stats. Manifests live in `datasets/manifests/`
(Git-tracked) while payloads stay Gitignored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import now_utc_iso


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class FileEntry(BaseModel):
    path: str  # relative to dataset root
    sha256: str
    bytes: int


class ImageEntry(BaseModel):
    image: FileEntry
    label: FileEntry | None = None  # None = background/negative image
    instances: int = 0
    classes: list[str] = Field(default_factory=list)  # class names present in this image


class DatasetManifest(BaseModel):
    """Immutable dataset snapshot: files + hashes + classes + provenance."""

    name: str
    created_at: str = Field(default_factory=now_utc_iso)
    format: str = "yolo"  # converter registry key used to produce this dataset
    root: str  # dataset root path at build time (informational)
    class_names: list[str] = Field(default_factory=list)  # index = class id
    split_seed: int = 42
    images: list[ImageEntry] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    notes: str = ""

    @property
    def total_images(self) -> int:
        return len(self.images)

    @property
    def total_instances(self) -> int:
        return sum(i.instances for i in self.images)


def build_manifest(
    root: Path,
    name: str,
    format: str = "yolo",
    class_names: list[str] | None = None,
    split_seed: int = 42,
    notes: str = "",
) -> DatasetManifest:
    """Walk a YOLO-format dataset root and hash everything.

    Accepted layouts:
        Split layout (post-split):  root/images/{train,val,test}/*.{ext}
        Split-first layout:         root/{train,val,test}/images/*.{ext}
                                    (labels at root/{split}/labels/ — ultralytics
                                    data.yaml convention)
        Flat interim layout:        root/images/*.{ext} + root/labels/*.txt

    The split name is "<split-dir-name>" in either split layout, "images" in the
    flat layout (files not yet assigned to a split).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    entries: list[ImageEntry] = []
    class_counts: dict[str, int] = {}
    per_split: dict[str, int] = {}

    # Layout detection: split-first ({split}/images) > images/{split} > flat.
    split_first_units: list[tuple[str, Path]] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if split_dir.name in {"images", "labels", "manifests"} or split_dir.name.startswith("."):
            continue
        if (split_dir / "images").is_dir():
            split_first_units.append((split_dir.name, split_dir / "images"))

    scan_units: list[tuple[str, Path]]
    img_dir = root / "images"
    if split_first_units:
        scan_units = split_first_units
    elif img_dir.is_dir():
        subdirs = sorted(p for p in img_dir.iterdir() if p.is_dir())
        scan_units = [(p.name, p) for p in subdirs] or [("images", img_dir)]
    else:
        raise FileNotFoundError(f"dataset root missing images/ directory: {root}")

    for split, split_dir in scan_units:
        for img_path in sorted(split_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in image_exts:
                continue
            rel_img = img_path.relative_to(root).as_posix()
            # NOTE: resolve label candidates against `root`, never the CWD.
            # split-first: {split}/labels/<stem>.txt | images-first: labels/{split}/<stem>.txt
            label_rel = Path(split) / "labels" / (img_path.stem + ".txt")
            if not (root / label_rel).is_file():
                label_rel = Path("labels") / split / (img_path.stem + ".txt")
            if not (root / label_rel).is_file():
                # flat layout fallback: labels/ directly next to images/
                label_rel = Path("labels") / (img_path.stem + ".txt")
            label_path = root / label_rel

            classes_here: list[str] = []
            instances = 0
            label_entry = None
            if label_path.is_file():
                for line in label_path.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue  # corrupt line; validator reports it
                    instances += 1
                    cname = class_names[cid] if class_names and 0 <= cid < len(class_names) else str(cid)
                    classes_here.append(cname)
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                label_entry = FileEntry(
                    path=label_rel.as_posix(),
                    sha256=sha256_file(label_path),
                    bytes=label_path.stat().st_size,
                )
            entries.append(
                ImageEntry(
                    image=FileEntry(path=rel_img, sha256=sha256_file(img_path), bytes=img_path.stat().st_size),
                    label=label_entry,
                    instances=instances,
                    classes=classes_here,
                )
            )
            per_split[split] = per_split.get(split, 0) + 1

    return DatasetManifest(
        name=name,
        format=format,
        root=str(root),
        class_names=list(class_names or []),
        split_seed=split_seed,
        images=entries,
        stats={
            "per_split": per_split,
            "class_counts": class_counts,
            "images_without_labels": sum(1 for e in entries if e.label is None),
            "total_instances": sum(e.instances for e in entries),
        },
        notes=notes,
    )


def save_manifest(manifest: DatasetManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

"""Stitch one episode's JPG frames into a contact sheet (10 thumbnails per row)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def episode_jpgs(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    return sorted(files, key=lambda p: p.name)


def contact_sheet(paths: list[Path], cols: int = 10, thumb_w: int = 160) -> Image.Image:
    if not paths:
        raise ValueError("no jpg frames")
    images = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = thumb_w / max(w, 1)
        images.append(im.resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.BILINEAR))
    cell_w = max(im.size[0] for im in images)
    cell_h = max(im.size[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 20, 20))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * cell_w, r * cell_h))
    return sheet


def write_sheet(folder: Path, out: Path | None = None, cols: int = 10) -> Path | None:
    paths = episode_jpgs(folder)
    if not paths:
        return None
    dest = out or (folder / "contact_sheet.jpg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(paths, cols=cols).save(dest, quality=85)
    return dest


def find_frame_dirs(root: Path) -> list[Path]:
    if (root / "frames").is_dir() and episode_jpgs(root / "frames"):
        return [root / "frames"]
    found = []
    for frames in sorted(root.glob("**/frames")):
        if frames.is_dir() and episode_jpgs(frames):
            found.append(frames)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cols", type=int, default=10)
    args = parser.parse_args()
    src = Path(args.src)
    dirs = find_frame_dirs(src)
    if src.is_dir() and episode_jpgs(src) and src.name == "frames":
        dirs = [src]
    if not dirs:
        raise SystemExit(f"no jpg frames under {src}")
    written = []
    for folder in dirs:
        dest = Path(args.out) if args.out and len(dirs) == 1 else folder / "contact_sheet.jpg"
        path = write_sheet(folder, dest, cols=args.cols)
        if path:
            written.append(path)
            print(path)
    if not written:
        raise SystemExit("no sheets written")


if __name__ == "__main__":
    main()

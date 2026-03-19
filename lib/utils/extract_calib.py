"""
Extract camera intrinsics from image EXIF or video metadata, save as calib.txt.

Strategy:
  --image: PIL EXIF → exiftool fallback → heuristic
  --video: ffprobe  → exiftool fallback → heuristic

Usage:
    python lib/utils/extract_calib.py --image photo.jpg  --output calib.txt
    python lib/utils/extract_calib.py --video video.mov  --output calib.txt
    python lib/utils/extract_calib.py --video video.mov  --verbose

Output format (same as SLAM calib.txt):
    fx fy cx cy

Requirements:
    Pillow    (pip install Pillow)               — for --image
    ffprobe   (https://ffmpeg.org/download.html) — for --video
    exiftool  optional, improves iPhone/Samsung support
              macOS: brew install exiftool
              Ubuntu: sudo apt install libimage-exiftool-perl
"""

import os
import json
import argparse
import subprocess

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Shared: exiftool fallback (works on both images and videos)
# ---------------------------------------------------------------------------

def parse_focal_lengths_exiftool(file_path):
    """
    Use exiftool to read focal length tags.
    Returns (fl_mm, fl_35mm), either can be None.
    """
    try:
        result = subprocess.run(
            ['exiftool', '-FocalLength', '-FocalLengthIn35mmFormat',
             '-CameraFocalLength', '-s3', file_path],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        return None, None  # exiftool not installed
    except subprocess.CalledProcessError:
        return None, None

    fl_mm, fl_35mm = None, None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        if fl_mm is None:
            fl_mm = val
        elif fl_35mm is None:
            fl_35mm = val

    # Single value >= 10mm is likely a 35mm equivalent, not actual focal length
    if fl_mm is not None and fl_35mm is None and fl_mm >= 10:
        fl_35mm, fl_mm = fl_mm, None

    return fl_mm, fl_35mm


# ---------------------------------------------------------------------------
# Image (PIL EXIF)
# ---------------------------------------------------------------------------

def get_exif(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data is None:
        return {}, img.size
    exif = {TAGS.get(tag_id, tag_id): val for tag_id, val in exif_data.items()}
    return exif, img.size


def parse_exif_focal_lengths(exif):
    def to_float(v):
        if v is None:
            return None
        if hasattr(v, 'numerator'):
            return float(v.numerator) / float(v.denominator)
        if isinstance(v, tuple):
            return v[0] / v[1]
        return float(v)

    return to_float(exif.get('FocalLength')), to_float(exif.get('FocalLengthIn35mmFilm'))


def extract_from_image(image_path):
    if not HAS_PIL:
        raise RuntimeError("Pillow is required: pip install Pillow")

    exif, (img_w, img_h) = get_exif(image_path)
    fl_mm, fl_35mm = parse_exif_focal_lengths(exif)

    if fl_mm is None and fl_35mm is None:
        fl_mm, fl_35mm = parse_focal_lengths_exiftool(image_path)

    return img_w, img_h, fl_mm, fl_35mm


def print_image_tags(exif):
    print("\n--- Image EXIF Tags ---")
    keys = ['Make', 'Model', 'FocalLength', 'FocalLengthIn35mmFilm',
            'ExifImageWidth', 'ExifImageHeight', 'DateTime']
    for k in keys:
        if k in exif:
            print(f"  {k}: {exif[k]}")
    print()


# ---------------------------------------------------------------------------
# Video (ffprobe + exiftool fallback)
# ---------------------------------------------------------------------------

def get_video_metadata(video_path):
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', '-show_format',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Install ffmpeg: https://ffmpeg.org/download.html")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")


def get_video_resolution(meta):
    for stream in meta.get('streams', []):
        if stream.get('codec_type') == 'video':
            w, h = stream.get('width'), stream.get('height')
            if w and h:
                return int(w), int(h)
    raise RuntimeError("No video stream found.")


def parse_ffprobe_focal_lengths(meta):
    all_tags = {}
    all_tags.update(meta.get('format', {}).get('tags', {}))
    for stream in meta.get('streams', []):
        all_tags.update(stream.get('tags', {}))

    fl_mm, fl_35mm = None, None
    for key, val in {k.lower(): v for k, v in all_tags.items()}.items():
        if 'focal_length' not in key:
            continue
        try:
            num = float(str(val).split('/')[0]) / float(str(val).split('/')[1]) \
                if '/' in str(val) else float(val)
        except (ValueError, TypeError, IndexError):
            continue
        if '35mm' in key:
            fl_35mm = num
        else:
            fl_mm = num

    return fl_mm, fl_35mm


def extract_from_video(video_path):
    meta = get_video_metadata(video_path)
    img_w, img_h = get_video_resolution(meta)
    fl_mm, fl_35mm = parse_ffprobe_focal_lengths(meta)

    if fl_mm is None and fl_35mm is None:
        fl_mm, fl_35mm = parse_focal_lengths_exiftool(video_path)

    return img_w, img_h, fl_mm, fl_35mm, meta


def print_video_tags(meta):
    print("\n--- Video Metadata Tags ---")
    fmt_tags = meta.get('format', {}).get('tags', {})
    if fmt_tags:
        print("  [format]")
        for k, v in fmt_tags.items():
            print(f"    {k}: {v}")
    for i, stream in enumerate(meta.get('streams', [])):
        tags = stream.get('tags', {})
        if tags:
            print(f"  [stream {i} — {stream.get('codec_type')}]")
            for k, v in tags.items():
                print(f"    {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# Shared: compute intrinsics
# ---------------------------------------------------------------------------

def compute_intrinsics(img_w, img_h, fl_mm, fl_35mm):
    cx, cy = img_w / 2.0, img_h / 2.0

    if fl_mm and fl_35mm:
        crop_factor = fl_35mm / fl_mm
        sensor_w = 36.0 / crop_factor
        sensor_h = 24.0 / crop_factor
        fx = fl_mm / sensor_w * img_w
        fy = fl_mm / sensor_h * img_h
        method = (f"focal_mm={fl_mm:.2f}mm + 35mm_equiv={fl_35mm}mm "
                  f"→ sensor={sensor_w:.2f}x{sensor_h:.2f}mm")
    elif fl_35mm:
        fx = fy = (fl_35mm / 36.0) * img_w
        method = f"35mm_equiv={fl_35mm}mm only (fx=fy assumed)"
    elif fl_mm:
        fx = fy = (fl_mm / 36.0) * img_w
        method = f"focal_mm={fl_mm:.2f}mm only, assumed full-frame sensor"
    else:
        fx = fy = (img_w ** 2 + img_h ** 2) ** 0.5
        method = "heuristic sqrt(w^2+h^2) — no focal length in metadata"

    return fx, fy, cx, cy, method


def save_calib(fx, fy, cx, cy, output_path):
    with open(output_path, 'w') as f:
        f.write(f"{fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract camera intrinsics from image EXIF or video metadata')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to image file (JPEG)')
    group.add_argument('--video', type=str, help='Path to video file (MP4/MOV/...)')
    parser.add_argument('--output',  type=str, default='calib.txt', help='Output calib.txt path')
    parser.add_argument('--verbose', action='store_true', help='Print all metadata tags')
    args = parser.parse_args()

    source = args.image or args.video
    if not os.path.exists(source):
        print(f"ERROR: File not found: {source}")
        exit(1)

    if args.image:
        if not HAS_PIL:
            print("ERROR: Pillow not installed. Run: pip install Pillow")
            exit(1)
        if args.verbose:
            exif, _ = get_exif(args.image)
            print_image_tags(exif)
        img_w, img_h, fl_mm, fl_35mm = extract_from_image(args.image)

    else:  # --video
        img_w, img_h, fl_mm, fl_35mm, meta = extract_from_video(args.video)
        if args.verbose:
            print_video_tags(meta)

    fx, fy, cx, cy, method = compute_intrinsics(img_w, img_h, fl_mm, fl_35mm)

    print(f"Source     : {source}")
    print(f"Resolution : {img_w} x {img_h} px")
    print(f"Method     : {method}")
    print(f"Intrinsics : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    save_calib(fx, fy, cx, cy, args.output)
    print(f"Saved      : {args.output}")

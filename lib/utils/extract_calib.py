"""
Extract camera intrinsics from video metadata and save as calib.txt.

Strategy:
  1. ffprobe  — reads standard QuickTime/MP4 tags
  2. exiftool — fallback for Apple-specific tags (iPhone focal length, etc.)
  3. Heuristic sqrt(w^2+h^2) if nothing found

Usage:
    python lib/utils/extract_calib.py --video path/to/video.mov --output calib.txt
    python lib/utils/extract_calib.py --video path/to/video.mov --verbose

Output format (same as SLAM calib.txt):
    fx fy cx cy

Requirements:
    ffmpeg/ffprobe  (https://ffmpeg.org/download.html)
    exiftool        optional but recommended for iPhone videos
                    macOS: brew install exiftool
                    Ubuntu: sudo apt install libimage-exiftool-perl
"""

import os
import json
import argparse
import subprocess


def get_video_metadata(video_path):
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
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


def parse_focal_lengths_exiftool(video_path):
    """
    Use exiftool to extract focal length tags — handles Apple QuickTime atoms
    that ffprobe misses (e.g. 'Camera Focal Length 35mm Equivalent').
    Returns (fl_mm, fl_35mm), either can be None.
    """
    try:
        result = subprocess.run(
            ['exiftool', '-FocalLength', '-FocalLengthIn35mmFormat',
             '-CameraFocalLength', '-s3', video_path],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        return None, None  # exiftool not installed, skip silently
    except subprocess.CalledProcessError:
        return None, None

    fl_mm, fl_35mm = None, None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # exiftool -s3 outputs bare values, one per -tag flag, in order
        # but values may look like "4.2 mm" or "27"
        try:
            val = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        if fl_mm is None:
            fl_mm = val
        elif fl_35mm is None:
            fl_35mm = val

    # If only one value came back and it looks like a 35mm equiv (>= 20mm typical)
    # treat it as fl_35mm, not fl_mm
    if fl_mm is not None and fl_35mm is None and fl_mm >= 10:
        fl_35mm, fl_mm = fl_mm, None

    return fl_mm, fl_35mm


def parse_focal_lengths(meta):
    """
    Try to extract focal length from video metadata tags.
    Returns (fl_mm, fl_35mm), either can be None.

    Common sources:
      - iPhone/iPad: com.apple.quicktime.camera.focal_length(.35mm_equivalent)
      - Some Android/GoPro: generic 'focal_length' tags in format or stream tags
    """
    fl_mm, fl_35mm = None, None

    all_tags = {}
    all_tags.update(meta.get('format', {}).get('tags', {}))
    for stream in meta.get('streams', []):
        all_tags.update(stream.get('tags', {}))

    # Normalize keys to lowercase for matching
    tags_lower = {k.lower(): v for k, v in all_tags.items()}

    for key, val in tags_lower.items():
        if 'focal_length' not in key:
            continue
        try:
            # Rational string "num/den" or plain float
            num = float(str(val).split('/')[0]) / float(str(val).split('/')[1]) \
                if '/' in str(val) else float(val)
        except (ValueError, TypeError, IndexError):
            continue

        if '35mm' in key:
            fl_35mm = num
        else:
            fl_mm = num

    return fl_mm, fl_35mm


def compute_intrinsics(img_w, img_h, fl_mm, fl_35mm):
    cx = img_w / 2.0
    cy = img_h / 2.0

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


def print_all_tags(meta):
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


def save_calib(fx, fy, cx, cy, output_path):
    with open(output_path, 'w') as f:
        f.write(f"{fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract camera intrinsics from video metadata')
    parser.add_argument('--video',   type=str, required=True, help='Path to video file (MP4/MOV/...)')
    parser.add_argument('--output',  type=str, default='calib.txt', help='Output calib.txt path')
    parser.add_argument('--verbose', action='store_true', help='Print all metadata tags')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        exit(1)

    meta = get_video_metadata(args.video)

    if args.verbose:
        print_all_tags(meta)

    img_w, img_h = get_video_resolution(meta)

    # Try ffprobe tags first, then exiftool as fallback
    fl_mm, fl_35mm = parse_focal_lengths(meta)
    if fl_mm is None and fl_35mm is None:
        fl_mm, fl_35mm = parse_focal_lengths_exiftool(args.video)

    fx, fy, cx, cy, method = compute_intrinsics(img_w, img_h, fl_mm, fl_35mm)

    print(f"Source     : {args.video}")
    print(f"Resolution : {img_w} x {img_h} px")
    print(f"Method     : {method}")
    print(f"Intrinsics : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    save_calib(fx, fy, cx, cy, args.output)
    print(f"Saved      : {args.output}")

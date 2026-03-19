"""
Extract camera intrinsics from video metadata (ffprobe) and save as calib.txt.

Usage:
    python lib/utils/extract_calib.py --video path/to/video.mp4 --output calib.txt
    python lib/utils/extract_calib.py --video path/to/video.mp4 --verbose

Output format (same as SLAM calib.txt):
    fx fy cx cy

Requirements:
    ffmpeg/ffprobe installed (https://ffmpeg.org/download.html)
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
    fl_mm, fl_35mm = parse_focal_lengths(meta)
    fx, fy, cx, cy, method = compute_intrinsics(img_w, img_h, fl_mm, fl_35mm)

    print(f"Source     : {args.video}")
    print(f"Resolution : {img_w} x {img_h} px")
    print(f"Method     : {method}")
    print(f"Intrinsics : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    save_calib(fx, fy, cx, cy, args.output)
    print(f"Saved      : {args.output}")

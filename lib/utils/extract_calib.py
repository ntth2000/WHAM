"""
Extract camera intrinsics from image EXIF metadata and save as calib.txt.

Usage:
    python lib/utils/extract_calib.py --image path/to/image.jpg --output calib.txt

Output format (same as SLAM calib.txt):
    fx fy cx cy
"""

import os
import argparse

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_exif(image_path):
    """Read raw EXIF tags from image."""
    if not HAS_PIL:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data is None:
        return {}, img.size

    exif = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        exif[tag] = value

    return exif, img.size


def parse_focal_length(exif):
    """Return focal length in mm from EXIF, or None."""
    fl = exif.get('FocalLength')
    if fl is None:
        return None
    if hasattr(fl, 'numerator'):
        return float(fl.numerator) / float(fl.denominator)
    if isinstance(fl, tuple):
        return fl[0] / fl[1]
    return float(fl)


def parse_focal_length_35mm(exif):
    """Return 35mm-equivalent focal length, or None."""
    fl = exif.get('FocalLengthIn35mmFilm')
    if fl is None:
        return None
    return float(fl)


def extract_intrinsics(image_path):
    """
    Extract (fx, fy, cx, cy) in pixels from image EXIF metadata.

    Strategy (in order of accuracy):
    1. focal length (mm) + 35mm equiv → derive sensor size → focal in pixels
    2. 35mm equivalent only → estimate focal in pixels (fx=fy, landscape assumed)
    3. Fallback: heuristic sqrt(w^2 + h^2)
    """
    exif, (img_w, img_h) = get_exif(image_path)

    cx = img_w / 2.0
    cy = img_h / 2.0

    fl_mm   = parse_focal_length(exif)
    fl_35mm = parse_focal_length_35mm(exif)

    if fl_mm and fl_35mm:
        crop_factor = fl_35mm / fl_mm
        sensor_w = 36.0 / crop_factor
        sensor_h = 24.0 / crop_factor
        fx = fl_mm / sensor_w * img_w
        fy = fl_mm / sensor_h * img_h
        method = f"focal_mm={fl_mm:.2f}mm, 35mm_equiv={fl_35mm}mm → sensor={sensor_w:.2f}x{sensor_h:.2f}mm"

    elif fl_35mm:
        fx = fy = (fl_35mm / 36.0) * img_w
        method = f"35mm_equiv={fl_35mm}mm only (fx=fy assumed)"

    else:
        fx = fy = (img_w ** 2 + img_h ** 2) ** 0.5
        method = "heuristic sqrt(w^2+h^2) — no EXIF focal length found"

    return fx, fy, cx, cy, img_w, img_h, method, exif


def print_exif_summary(exif):
    keys = ['Make', 'Model', 'FocalLength', 'FocalLengthIn35mmFilm',
            'ExifImageWidth', 'ExifImageHeight', 'DateTime']
    print("\n--- EXIF Summary ---")
    for k in keys:
        if k in exif:
            print(f"  {k}: {exif[k]}")
    print()


def save_calib(fx, fy, cx, cy, output_path):
    with open(output_path, 'w') as f:
        f.write(f"{fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract camera intrinsics from image EXIF')
    parser.add_argument('--image',  type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='calib.txt', help='Output calib.txt path')
    parser.add_argument('--verbose', action='store_true', help='Print EXIF details')
    args = parser.parse_args()

    if not HAS_PIL:
        print("ERROR: Pillow not installed. Run: pip install Pillow")
        exit(1)

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        exit(1)

    fx, fy, cx, cy, img_w, img_h, method, exif = extract_intrinsics(args.image)

    if args.verbose:
        print_exif_summary(exif)

    print(f"Image size : {img_w} x {img_h} px")
    print(f"Method     : {method}")
    print(f"Intrinsics : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    save_calib(fx, fy, cx, cy, args.output)
    print(f"Saved      : {args.output}")

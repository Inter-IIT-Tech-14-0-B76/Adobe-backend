import argparse
import os
import sys
import requests


INFER_HOST = "https://unpj0fg9si7yet-7860.proxy.runpod.net/infer"
CONVERT_HOST = "https://unpj0fg9si7yet-7860.proxy.runpod.net/convert"


def test_infer(infer_url, key, images_dir, out_mp4):
    """Test /infer endpoint."""
    images = []
    for i in range(4):
        img_path = os.path.join(images_dir, f"view{i}.png")
        if not os.path.exists(img_path):
            print(f"ERROR: Missing {img_path}")
            sys.exit(1)
        images.append(img_path)

    files = []
    for i, path in enumerate(images):
        files.append((f"view{i}", (f"view{i}.png", open(path, "rb"), "image/png")))

    print("\n=== TESTING /infer ===")
    print(f"[TEST] POST {infer_url}")
    print(f"[TEST] key = {key}")
    print(f"[TEST] images = {images}")

    try:
        r = requests.post(
            infer_url,
            data={"key": key},
            files=files,
            timeout=1200,
        )
    except Exception as e:
        print(f"[ERROR] HTTP error calling /infer: {e}")
        sys.exit(1)

    if r.status_code != 200:
        print(f"[ERROR] /infer returned {r.status_code}")
        try:
            print(r.json())
        except Exception:
            print(r.text)
        sys.exit(1)

    # Save result MP4
    with open(out_mp4, "wb") as f:
        f.write(r.content)
    print(f"[TEST] Saved MP4 to {out_mp4}")


def test_convert(convert_url, key, out_glb):
    """Test /convert endpoint using the key (no file upload)."""

    print("\n=== TESTING /convert ===")
    print(f"[TEST] POST {convert_url}")
    print(f"[TEST] key = {key}")

    try:
        r = requests.post(
            convert_url,
            data={"key": key},
            timeout=1800,
        )
    except Exception as e:
        print(f"[ERROR] HTTP error calling /convert: {e}")
        sys.exit(1)

    if r.status_code != 200:
        print(f"[ERROR] /convert returned {r.status_code}")
        try:
            print(r.json())
        except Exception:
            print(r.text)
        sys.exit(1)

    # Save returned GLB
    with open(out_glb, "wb") as f:
        f.write(r.content)
    print(f"[TEST] Saved GLB to {out_glb}")


def main():
    parser = argparse.ArgumentParser(description="Test LGM /infer and /convert APIs")
    parser.add_argument(
        "--infer_url", default="http://localhost:5000/infer", help="Infer endpoint URL"
    )
    parser.add_argument(
        "--convert_url",
        default="http://localhost:5010/convert",
        help="Convert endpoint URL",
    )
    parser.add_argument("--key", required=True, help="Unique key for this test job")
    parser.add_argument(
        "--images_dir", required=True, help="Directory containing view0.png..view3.png"
    )
    parser.add_argument(
        "--mp4_out", default=None, help="Where to save MP4 (default: <key>.mp4)"
    )
    parser.add_argument(
        "--glb_out", default=None, help="Where to save GLB (default: <key>.glb)"
    )

    args = parser.parse_args()

    mp4_out = args.mp4_out or f"{args.key}.mp4"
    glb_out = args.glb_out or f"{args.key}.glb"

    # Step 1: Call /infer → produce MP4 + PLY
    test_infer(args.infer_url, args.key, args.images_dir, mp4_out)

    # Step 2: Call /convert → produce GLB from saved PLY
    test_convert(args.convert_url, args.key, glb_out)


if __name__ == "__main__":
    main()

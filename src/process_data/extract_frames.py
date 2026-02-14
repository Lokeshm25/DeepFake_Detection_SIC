# src/data/extract_frames.py
import cv2
from pathlib import Path
import argparse

def sample_n_frames(video_path, out_dir, n=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(str(video_path))
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("Skipping (no frames):", video_path)
        return 0
    indices = [int(i * total / n) for i in range(n)]
    saved = 0
    for i, idx in enumerate(indices):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vidcap.read()
        if not ret:
            continue
        fname = out_dir / f"{Path(video_path).stem}_frame_{i:02d}.jpg"
        cv2.imwrite(str(fname), frame)
        saved += 1
    vidcap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder with videos")
    parser.add_argument("--out", required=True, help="Output folder for frames")
    parser.add_argument("--n", type=int, default=8, help="Frames per video")
    parser.add_argument("--ext", default="mp4", help="Video extension to search (mp4)")
    args = parser.parse_args()

    src = Path(args.src)
    vids = list(src.glob(f"*.{args.ext}"))
    print("Found", len(vids), "videos in", src)
    for v in vids:
        n = sample_n_frames(v, Path(args.out) / v.stem, n=args.n)
        if n > 0:
            print("Saved", n, "frames for", v.stem)

if __name__ == "__main__":
    main()

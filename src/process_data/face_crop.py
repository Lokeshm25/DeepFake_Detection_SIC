# src/data/face_crop.py
import argparse
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN   # MTCNN fallback
# If you prefer RetinaFace, you can plug it here (RetinaFace install steps differ)
import numpy as np

def crop_faces_from_frames(frames_root, out_root, size=224, device='cuda'):
    mtcnn = MTCNN(keep_all=False, device=device)
    frames_root = Path(frames_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    frames = list(frames_root.glob("*/*.jpg"))  # nested: <video_stem>/frame_xxx.jpg
    print("Total frames to process:", len(frames))
    success = 0
    for fp in frames:
        try:
            img = Image.open(fp).convert('RGB')
        except Exception as e:
            print("Error opening", fp, e); continue
        face = mtcnn(img)
        if face is None:
            continue
        arr = face.permute(1,2,0).int().numpy().astype('uint8')
        out_dir = out_root / fp.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (fp.stem + "_face.jpg")
        Image.fromarray(arr).save(out_path)
        success += 1
    print("Saved face crops:", success)
    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    crop_faces_from_frames(args.frames_root, args.out_root, size=args.size, device=args.device)

if __name__ == "__main__":
    main()

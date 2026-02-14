# src/data/augmentations.py
import subprocess
from pathlib import Path

def reencode_video(input_path, output_path, qscale=23):
    """
    Re-encode input with libx264 using a target qp.
    In ffmpeg qscale lower = higher quality; choose qscale ~ 17-28 as needed.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", str(qscale),
        "-preset", "veryfast", "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def reencode_h264_qf(input_path, output_path, qf=85):
    """
    Convert 'qf' ~ quality fraction to a CRF approx.
    qf 85 -> high quality ~ crf 18, qf 75 -> crf 23 -- approximate mapping
    """
    # mapping heuristic
    qf_to_crf = {85: 18, 75: 23}
    crf = qf_to_crf.get(int(qf), 23)
    return reencode_video(input_path, output_path, qscale=crf)

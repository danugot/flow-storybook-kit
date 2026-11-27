#!/usr/bin/env python3
"""Audit rendered MP4 videos against vertical mobile feed delivery standards.

Runs ffprobe + ffmpeg (ebur128 loudness + signalstats YAVG) and checks:
- 1080x1920 vertical full-screen resolution
- FPS >= 25 (30fps recommended)
- Video bitrate (6-14 Mbps to avoid dark-scene banding)
- Integrated loudness (-16.5..-13.0 LUFS), LRA (<= 8.5 LU), True Peak (<= -1.0 dBFS)
- Mean YAVG brightness (>= 60), opening frame YAVG (>= 35), and dark frames (< 30)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TARGETS: Dict[str, Any] = {
    "width": 1080,
    "height": 1920,
    "min_fps": 25.0,
    "recommended_fps": 30.0,
    "bitrate_mbps_min": 6.0,
    "bitrate_mbps_max": 14.0,
    "lufs_min": -16.5,
    "lufs_max": -13.0,
    "lra_max": 8.5,
    "true_peak_max_dbfs": -1.0,
    "yavg_min": 60.0,
    "yavg_ideal_min": 80.0,
}


def parse_fps(rate_str: str) -> float:
    """Parse ffprobe rational frame rate like '30/1' or '24000/1001'."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        try:
            den = float(den_s)
            return round(float(num_s) / den, 3) if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return round(float(rate_str), 3)
    except ValueError:
        return 0.0


def run_ffprobe(video_path: str) -> Dict[str, Any]:
    """Extract stream & container metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})

    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)
    v_bitrate = v_stream.get("bit_rate")
    if v_bitrate:
        bitrate_mbps = round(float(v_bitrate) / 1_000_000.0, 2)
    elif fmt.get("bit_rate"):
        bitrate_mbps = round(float(fmt["bit_rate"]) / 1_000_000.0, 2)
    elif duration > 0 and fmt.get("size"):
        bitrate_mbps = round((float(fmt["size"]) * 8.0 / duration) / 1_000_000.0, 2)
    else:
        bitrate_mbps = 0.0

    return {
        "width": int(v_stream.get("width") or 0),
        "height": int(v_stream.get("height") or 0),
        "fps": parse_fps(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate") or "0"),
        "duration_sec": round(duration, 2),
        "bitrate_mbps": bitrate_mbps,
        "video_codec": v_stream.get("codec_name", ""),
        "pix_fmt": v_stream.get("pix_fmt", ""),
        "has_audio": bool(a_stream),
        "audio_sample_rate": int(a_stream.get("sample_rate") or 0) if a_stream else 0,
    }


def run_ebur128(video_path: str) -> Dict[str, Optional[float]]:
    """Measure Integrated LUFS, LRA, and True Peak via ffmpeg ebur128."""
    cmd = [
        "ffmpeg", "-nostats", "-i", video_path,
        "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    text = res.stderr
    summary_idx = text.rfind("Summary:")
    tail = text[summary_idx:] if summary_idx != -1 else text

    def extract(pat: str) -> Optional[float]:
        m = re.search(pat, tail)
        return float(m.group(1)) if m else None

    return {
        "integrated_lufs": extract(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS"),
        "lra_lu": extract(r"LRA:\s*(-?\d+(?:\.\d+)?)\s*LU"),
        "true_peak_dbfs": extract(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS"),
    }


def run_signalstats(video_path: str, sample_fps: float) -> Dict[str, Any]:
    """Sample frame YAVG values across the video using ffmpeg signalstats."""
    vf = f"fps={sample_fps},signalstats,metadata=print:file=-"
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-an", "-vf", vf, "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    yavgs = [float(val) for val in re.findall(r"lavfi\.signalstats\.YAVG=(\d+(?:\.\d+)?)", res.stdout)]
    if not yavgs:
        return {
            "samples": 0,
            "yavg_mean": None,
            "yavg_min": None,
            "yavg_max": None,
            "first_yavg": None,
            "dark_samples": 0,
        }
    return {
        "samples": len(yavgs),
        "yavg_mean": round(statistics.mean(yavgs), 1),
        "yavg_min": round(min(yavgs), 1),
        "yavg_max": round(max(yavgs), 1),
        "first_yavg": round(yavgs[0], 1),
        "dark_samples": sum(1 for y in yavgs if y < 30.0),
    }


def evaluate(
    probe: Dict[str, Any],
    audio: Dict[str, Optional[float]],
    luma: Dict[str, Any],
    targets: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Compare measured metrics against mobile feed targets and build repair hints."""
    checks: List[Dict[str, str]] = []
    fixes: List[str] = []

    def add(name: str, value: str, target: str, status: str, note: str) -> None:
        checks.append({"metric": name, "value": value, "target": target, "status": status, "note": note})

    # 1. Resolution
    w, h = probe["width"], probe["height"]
    exp_w, exp_h = int(targets["width"]), int(targets["height"])
    if w == exp_w and h == exp_h:
        add("Resolution", f"{w}x{h}", f"{exp_w}x{exp_h}", "PASS", "Full-bleed vertical frame")
    elif h > w and w >= 720:
        add("Resolution", f"{w}x{h}", f"{exp_w}x{exp_h}", "WARN", "Vertical ratio OK but below 1080x1920")
    else:
        add("Resolution", f"{w}x{h}", f"{exp_w}x{exp_h}", "FAIL", "Not vertical 1080x1920 feed spec")
        fixes.append(f"scale={exp_w}:{exp_h}:force_original_aspect_ratio=increase,crop={exp_w}:{exp_h}")

    # 2. FPS
    fps = probe["fps"]
    rec_fps = float(targets["recommended_fps"])
    min_fps = float(targets["min_fps"])
    if fps >= rec_fps:
        add("Frame Rate", f"{fps} fps", f">= {rec_fps:.0f} fps", "PASS", "Smooth mobile playback")
    elif fps >= min_fps:
        add("Frame Rate", f"{fps} fps", f"{rec_fps:.0f} fps", "WARN", "Slightly below 30fps target")
    else:
        add("Frame Rate", f"{fps} fps", f">= {min_fps:.0f} fps", "WARN", "Low FPS may judder after re-encoding")

    # 3. Bitrate
    br = probe["bitrate_mbps"]
    br_min = float(targets["bitrate_mbps_min"])
    br_max = float(targets["bitrate_mbps_max"])
    if br_min <= br <= br_max:
        add("Video Bitrate", f"{br} Mbps", f"{br_min:.0f}-{br_max:.0f} Mbps", "PASS", "Clean gradient headroom")
    elif br < 5.0:
        add("Video Bitrate", f"{br} Mbps", f">= {br_min:.0f} Mbps", "FAIL", "Too low; dark scenes risk banding")
    elif br > br_max:
        add("Video Bitrate", f"{br} Mbps", f"<= {br_max:.0f} Mbps", "WARN", "High bitrate; platform may compress aggressively")
    else:
        add("Video Bitrate", f"{br} Mbps", f"{br_min:.0f}-{br_max:.0f} Mbps", "WARN", "Slightly below recommended bitrate")

    # 4. Audio loudness
    if not probe["has_audio"]:
        add("Audio Track", "missing", "48kHz -15 LUFS", "FAIL", "No audio stream present")
    else:
        lufs = audio.get("integrated_lufs")
        lra = audio.get("lra_lu")
        tp = audio.get("true_peak_dbfs")
        lufs_min = float(targets["lufs_min"])
        lufs_max = float(targets["lufs_max"])
        if lufs is None:
            add("Integrated LUFS", "unknown", f"{lufs_min}..{lufs_max} LUFS", "WARN", "Failed to parse ebur128 LUFS")
        elif lufs_min <= lufs <= lufs_max:
            add("Integrated LUFS", f"{lufs} LUFS", f"{lufs_min}..{lufs_max} LUFS", "PASS", "Phone speaker loudness OK")
        elif lufs < -18.5:
            add("Integrated LUFS", f"{lufs} LUFS", "-15 LUFS", "FAIL", "Too quiet for mobile feeds")
            fixes.append("loudnorm=I=-15:TP=-1.5:LRA=7")
        else:
            add("Integrated LUFS", f"{lufs} LUFS", f"{lufs_min}..{lufs_max} LUFS", "WARN", "Consider converging near -15 LUFS")
            fixes.append("loudnorm=I=-15:TP=-1.5:LRA=7")

        lra_max = float(targets["lra_max"])
        if lra is not None:
            if lra <= lra_max:
                add("Loudness Range", f"{lra} LU", f"<= {lra_max} LU", "PASS", "Consistent speech level")
            elif lra <= 11.5:
                add("Loudness Range", f"{lra} LU", f"<= {lra_max} LU", "WARN", "Slightly wide dynamic range")
            else:
                add("Loudness Range", f"{lra} LU", f"<= {lra_max} LU", "FAIL", "Dynamic range too wide")

        tp_max = float(targets["true_peak_max_dbfs"])
        if tp is not None:
            if tp <= tp_max:
                add("True Peak", f"{tp} dBFS", f"<= {tp_max} dBFS", "PASS", "AAC headroom safe")
            else:
                add("True Peak", f"{tp} dBFS", f"<= {tp_max} dBFS", "FAIL", "Clipping risk on transcode")

    # 5. Brightness
    ym = luma.get("yavg_mean")
    y_min_ok = float(targets["yavg_min"])
    y_ideal = float(targets["yavg_ideal_min"])
    if ym is None:
        add("Mean Luma YAVG", "unknown", f">= {y_min_ok}", "WARN", "No luma frames sampled")
    elif ym >= y_ideal:
        add("Mean Luma YAVG", f"{ym}", f">= {y_ideal}", "PASS", "Clear visibility on phone screens")
    elif ym >= y_min_ok:
        add("Mean Luma YAVG", f"{ym}", f"{y_ideal}-120", "WARN", "Usable but slightly dim")
    else:
        add("Mean Luma YAVG", f"{ym}", f">= {y_min_ok}", "FAIL", "Too dark overall")
        fixes.append("curves=all='0/0 0.15/0.20 0.4/0.52 0.7/0.80 1/1'")

    first_y = luma.get("first_yavg")
    if first_y is not None:
        if first_y < 35.0:
            add("Opening Frame", f"{first_y}", ">= 35.0", "WARN", "Opening frame too dark")
        else:
            add("Opening Frame", f"{first_y}", ">= 35.0", "PASS", "Opening frame clearly visible")

    dark_cnt = int(luma.get("dark_samples") or 0)
    if dark_cnt > 1:
        add("Dark Frames (<30)", f"{dark_cnt}", "<= 1", "WARN", f"Min YAVG={luma.get('yavg_min')}; inspect black gaps")
    else:
        add("Dark Frames (<30)", f"{dark_cnt}", "<= 1", "PASS", "No prolonged black gaps")

    return checks, list(dict.fromkeys(fixes))


def load_targets(config_path: Optional[str]) -> Dict[str, Any]:
    targets = dict(DEFAULT_TARGETS)
    if not config_path:
        return targets
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    override = raw.get("mobile_feed_targets", raw) if isinstance(raw, dict) else {}
    if isinstance(override, dict):
        targets.update(override)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MP4 mobile feed readiness.")
    parser.add_argument("video", help="Path to MP4 file")
    parser.add_argument("--config", default=None, help="Optional JSON file overriding mobile_feed_targets")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Luma sampling rate in fps (default: 1.0)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: file not found: {args.video}", file=sys.stderr)
        return 2

    targets = load_targets(args.config)
    probe = run_ffprobe(args.video)
    audio = run_ebur128(args.video) if probe["has_audio"] else {}
    luma = run_signalstats(args.video, args.sample_fps)
    checks, fixes = evaluate(probe, audio, luma, targets)

    fail_cnt = sum(1 for c in checks if c["status"] == "FAIL")
    warn_cnt = sum(1 for c in checks if c["status"] == "WARN")
    pass_cnt = len(checks) - fail_cnt - warn_cnt

    if args.json:
        print(json.dumps({
            "video": args.video,
            "probe": probe,
            "audio": audio,
            "luma": luma,
            "checks": checks,
            "fixes": fixes,
            "pass_count": pass_cnt,
            "warn_count": warn_cnt,
            "fail_count": fail_cnt,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"=== Feed Delivery Audit: {os.path.basename(args.video)} ({probe['duration_sec']}s) ===")
        print(f"{'Metric':<18} | {'Measured':<16} | {'Target':<18} | {'Status':<6} | Note")
        print("-" * 92)
        for c in checks:
            print(f"{c['metric']:<18} | {c['value']:<16} | {c['target']:<18} | {c['status']:<6} | {c['note']}")
        print("-" * 92)
        print(f"Summary: {pass_cnt} PASS, {warn_cnt} WARN, {fail_cnt} FAIL")
        if fixes or fail_cnt > 0:
            vf_parts = ["fps=30"]
            af_parts = []
            for fx in fixes:
                if fx.startswith("loudnorm"):
                    af_parts.append(fx)
                else:
                    vf_parts.append(fx)
            vf_str = ",".join(vf_parts)
            af_str = f'-af "{",".join(af_parts)}" ' if af_parts else ""
            out_name = os.path.splitext(args.video)[0] + "_feed_fixed.mp4"
            print("\nSuggested fix command:")
            print(
                f'ffmpeg -y -i "{args.video}" -vf "{vf_str}" {af_str}'
                f'-c:v libx264 -preset slow -crf 17 -profile:v high -pix_fmt yuv420p '
                f'-c:a aac -b:a 192k -ar 48000 "{out_name}"'
            )

    return 2 if fail_cnt > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

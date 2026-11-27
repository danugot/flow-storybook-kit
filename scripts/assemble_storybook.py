#!/usr/bin/env python3
"""Assemble 1080x1920 @ 30fps storybook episodes from paginated illustrations and clips.

Features:
- Sub-pixel Ken Burns on 1.5x-oversampled LANCZOS canvases via cv2.warpAffine
  (eliminates ffmpeg zoompan integer-step jitter / 7.5Hz flicker).
- Optional animated page clips (1:1 24fps->30fps or resampled), seamlessly held on
  the exact final frame without double color grading.
- Universal ffmpeg audio ingestion for any sample rate, bit depth, or container
  (.wav, .mp3, .m4a, .flac, .ogg), mixed in NumPy with smooth BGM ducking.
- 2-pass EBU R128 loudness normalization with silent-track safety fallback.
- Data-driven bilingual or single-language subtitle pills and optional recap card.

Python 3.9+ compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLE_RATE = 48000
TARGET_W = 1080
TARGET_H = 1920
OVERSAMPLE_W = 1620
OVERSAMPLE_H = 2880
FPS = 30

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PRESET_BOTTOM_Y = {
    "shorts": 1420,
    "douyin": 1440,
    "xiaohongshu": 1365,
}

EN_FONT_CANDIDATES: List[Tuple[str, int]] = [
    ("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("C:/Windows/Fonts/arialbd.ttf", 0),
    ("C:/Windows/Fonts/segoeuib.ttf", 0),
]

ZH_FONT_CANDIDATES: List[Tuple[str, int]] = [
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("C:/Windows/Fonts/msyhbd.ttc", 0),
    ("C:/Windows/Fonts/msyh.ttc", 0),
    ("C:/Windows/Fonts/simhei.ttf", 0),
]

DEFAULT_NARRATOR_STYLE: Dict[str, Any] = {
    "fill": (255, 224, 130, 255),
    "outline": (255, 214, 102, 225),
    "tag": None,
    "tag_bg": None,
    "tag_fg": None,
}

FALLBACK_SPEAKER_PALETTE: List[Dict[str, Any]] = [
    {
        "fill": (150, 222, 255, 255),
        "outline": (130, 206, 255, 225),
        "tag_bg": (130, 206, 255, 248),
        "tag_fg": (18, 28, 42, 255),
    },
    {
        "fill": (255, 196, 112, 255),
        "outline": (255, 178, 92, 225),
        "tag_bg": (255, 186, 98, 248),
        "tag_fg": (42, 20, 8, 255),
    },
    {
        "fill": (166, 240, 200, 255),
        "outline": (152, 232, 188, 225),
        "tag_bg": (152, 232, 188, 248),
        "tag_fg": (16, 36, 24, 255),
    },
    {
        "fill": (255, 232, 138, 255),
        "outline": (255, 212, 92, 225),
        "tag_bg": (255, 212, 92, 248),
        "tag_fg": (42, 28, 6, 255),
    },
    {
        "fill": (214, 186, 255, 255),
        "outline": (194, 160, 248, 225),
        "tag_bg": (194, 160, 248, 248),
        "tag_fg": (28, 16, 44, 255),
    },
]


def load_font(explicit_path: Optional[str], candidates: Sequence[Tuple[str, int]], size: int):
    """Load a TrueType/TTC font across macOS, Linux, and Windows with safe fallback."""
    trials: List[Tuple[str, int]] = []
    if explicit_path:
        trials.append((explicit_path, 0))
    trials.extend(candidates)
    for path, idx in trials:
        if not path or not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _as_rgba(val: Any, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) in (3, 4):
        nums = [int(x) for x in val]
        if len(nums) == 3:
            nums.append(255)
        return (nums[0], nums[1], nums[2], nums[3])
    return default


def resolve_speaker_style(speaker: str, speakers_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve badge and text styling from script.json or deterministic fallback palette."""
    if speaker in speakers_cfg and isinstance(speakers_cfg[speaker], dict):
        raw = speakers_cfg[speaker]
        fill = _as_rgba(raw.get("fill"), DEFAULT_NARRATOR_STYLE["fill"])
        outline = _as_rgba(raw.get("outline"), (fill[0], fill[1], fill[2], 225))
        tag = raw.get("tag")
        tag_bg = _as_rgba(raw.get("tag_bg"), (outline[0], outline[1], outline[2], 248)) if tag else None
        tag_fg = _as_rgba(raw.get("tag_fg"), (20, 24, 34, 255)) if tag else None
        return {
            "fill": fill,
            "outline": outline,
            "tag": tag,
            "tag_bg": tag_bg,
            "tag_fg": tag_fg,
        }

    if speaker.strip().lower() == "narrator":
        return dict(DEFAULT_NARRATOR_STYLE)

    idx = sum(ord(ch) for ch in speaker) % len(FALLBACK_SPEAKER_PALETTE)
    pal = FALLBACK_SPEAKER_PALETTE[idx]
    return {
        "fill": pal["fill"],
        "outline": pal["outline"],
        "tag": speaker,
        "tag_bg": pal["tag_bg"],
        "tag_fg": pal["tag_fg"],
    }


def find_page_image(pages_dir: str, page_id: str) -> str:
    for ext in IMAGE_EXTS:
        p = os.path.join(pages_dir, page_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No page image found for {page_id} in {pages_dir}")


def find_line_audio(lines_dir: str, line_id: str, ep_dir: str, line_obj: Dict[str, Any]) -> str:
    explicit = line_obj.get("audio_file")
    if explicit:
        cand = explicit if os.path.isabs(explicit) else os.path.join(ep_dir, explicit)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"Explicit line audio_file not found: {cand}")

    for stem in (f"line_{line_id}", str(line_id)):
        for ext in AUDIO_EXTS:
            cand = os.path.join(lines_dir, stem + ext)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f"No audio file found for line_id={line_id} in {lines_dir}")


def find_bgm_file(ep_dir: str, script: Dict[str, Any], bgm_arg: Optional[str]) -> Optional[str]:
    if bgm_arg is not None:
        if bgm_arg.strip().lower() in ("", "none", "off"):
            return None
        cand = bgm_arg if os.path.isabs(bgm_arg) else os.path.join(ep_dir, bgm_arg)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"--bgm file not found: {cand}")

    cfg_bgm = script.get("bgm")
    if isinstance(cfg_bgm, str) and cfg_bgm.strip():
        cand = cfg_bgm if os.path.isabs(cfg_bgm) else os.path.join(ep_dir, cfg_bgm)
        if os.path.exists(cand):
            return cand

    audio_dir = os.path.join(ep_dir, "audio")
    for stem in ("bgm", "lullaby_bed"):
        for ext in AUDIO_EXTS:
            cand = os.path.join(audio_dir, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def probe_duration(path: str) -> float:
    res = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(res.stdout.strip())


def decode_audio_ffmpeg(path: str, channels: int = 1, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file/rate/bit-depth into float32 NumPy samples at target rate."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(sample_rate),
        "pipe:1",
    ]
    res = subprocess.run(cmd, check=True, capture_output=True)
    arr = np.frombuffer(res.stdout, dtype=np.float32)
    if channels == 2:
        usable = (len(arr) // 2) * 2
        return arr[:usable].reshape((-1, 2))
    return arr


def prepare_leveled_bgm(raw_path: str, out_wav: str) -> None:
    """Apply gentle low-pass warmth and compression so BGM sits softly under dialogue."""
    flt = (
        "highpass=f=60,"
        "lowpass=f=4400,"
        "acompressor=threshold=-26dB:ratio=4.5:attack=25:release=350:makeup=5,"
        "alimiter=limit=0.75"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", raw_path,
            "-af", flt,
            "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le",
            out_wav,
        ],
        check=True,
        capture_output=True,
    )


def _is_finite_float(val: Any) -> bool:
    try:
        num = float(val)
        return math.isfinite(num)
    except (TypeError, ValueError):
        return False


def build_audio_and_timeline(
    ep_dir: str,
    work_dir: str,
    script: Dict[str, Any],
    bgm_arg: Optional[str] = None,
    target_lufs: float = -15.0,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]], float, float]:
    lines_dir = os.path.join(ep_dir, "audio", "lines")
    audio_out_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    dissolve_sec = 0.45
    pages = script.get("pages", [])
    if not pages:
        raise ValueError("script.json must contain at least one page in 'pages'.")

    page_specs: List[Dict[str, Any]] = []
    cues: List[Dict[str, Any]] = []

    cursor = 0.0
    for pi, p in enumerate(pages):
        pid = p["page_id"]
        pre_pad = 0.55 if pi == 0 else (dissolve_sec + 0.35)
        post_pad = float(p.get("hold_tail_sec", 0.45))
        if pi < len(pages) - 1:
            post_pad += dissolve_sec

        local_t = pre_pad
        page_start = cursor - (dissolve_sec if pi > 0 else 0.0)
        page_lines = p.get("lines") or []

        if page_lines:
            for li, ln in enumerate(page_lines):
                lid = str(ln["line_id"])
                aud_p = find_line_audio(lines_dir, lid, ep_dir, ln)
                samples = decode_audio_ffmpeg(aud_p, channels=1, sample_rate=SAMPLE_RATE)
                dur = len(samples) / float(SAMPLE_RATE)

                abs_t0 = (page_start if pi > 0 else 0.0) + local_t
                abs_t1 = abs_t0 + dur
                cues.append({
                    "line_id": lid,
                    "page_id": pid,
                    "speaker": ln.get("speaker", "Narrator"),
                    "en": ln.get("en", ""),
                    "zh": ln.get("zh", ""),
                    "t0": round(abs_t0, 3),
                    "t1": round(abs_t1, 3),
                    "samples": samples,
                    "sfx": ln.get("sfx"),
                    "dur": round(dur, 3),
                })
                local_t += dur
                if li < len(page_lines) - 1:
                    local_t += 0.48
            local_t += post_pad
        else:
            silent_dur = float(p.get("duration_sec", 3.0))
            local_t = max(silent_dur, pre_pad + post_pad)

        if p.get("animated"):
            min_anim = float(p.get("clip_duration_target", 5)) * 1.12 + 0.8
            if local_t < min_anim:
                local_t = min_anim

        page_specs.append({
            "page_id": pid,
            "animated": bool(p.get("animated", False)),
            "ken_burns": p.get("ken_burns", "in"),
            "duration": round(local_t, 3),
        })
        cursor = (page_start if pi > 0 else 0.0) + local_t

    total_dur = max(0.5, round(cursor, 3))
    n_total = int(math.ceil(total_dur * SAMPLE_RATE))

    voice_buf = np.zeros(n_total, dtype=np.float32)
    foley_buf = np.zeros(n_total, dtype=np.float32)
    speech_env = np.zeros(n_total, dtype=np.float32)

    for cue in cues:
        samples = cue["samples"]
        idx0 = int(round(cue["t0"] * SAMPLE_RATE))
        dst0 = max(0, idx0)
        src0 = max(0, -idx0)
        copy_len = min(len(samples) - src0, n_total - dst0)
        if copy_len > 0:
            voice_buf[dst0 : dst0 + copy_len] += samples[src0 : src0 + copy_len]
            speech_env[dst0 : dst0 + copy_len] = 1.0

    for cue in cues:
        sfx = cue.get("sfx")
        if not sfx or not isinstance(sfx, dict) or not sfx.get("file"):
            continue
        sfx_rel = str(sfx["file"])
        sfx_path = sfx_rel if os.path.isabs(sfx_rel) else os.path.join(ep_dir, sfx_rel)
        if not os.path.exists(sfx_path):
            print(f"   [WARN] Missing sfx file: {sfx_path}")
            continue
        sfx_samples = decode_audio_ffmpeg(sfx_path, channels=1, sample_rate=SAMPLE_RATE)
        s_idx0 = int(round((cue["t0"] + float(sfx.get("offset_sec", 0.0))) * SAMPLE_RATE))
        s_gain = float(sfx.get("gain", 0.30))
        dst0 = max(0, s_idx0)
        src0 = max(0, -s_idx0)
        copy_len = min(len(sfx_samples) - src0, n_total - dst0)
        if copy_len > 0:
            foley_buf[dst0 : dst0 + copy_len] += sfx_samples[src0 : src0 + copy_len] * s_gain

    base_mono = voice_buf + foley_buf
    bgm_src = find_bgm_file(ep_dir, script, bgm_arg)

    if bgm_src:
        bgm_warm = os.path.join(audio_out_dir, "bgm_warm_leveled.wav")
        prepare_leveled_bgm(bgm_src, bgm_warm)
        music_st = decode_audio_ffmpeg(bgm_warm, channels=2, sample_rate=SAMPLE_RATE)
        if len(music_st) == 0:
            bgm_src = None

    if bgm_src:
        gain_under_voice = 0.095
        gain_free = 0.24
        step = SAMPLE_RATE // 100  # 10 ms bins
        n_bins = max(1, int(math.ceil(n_total / float(step))))
        pad_len = n_bins * step - n_total
        env_padded = np.pad(speech_env, (0, pad_len)) if pad_len > 0 else speech_env
        env_100 = env_padded.reshape((n_bins, step)).max(axis=1)

        dt = 0.01
        atk = math.exp(-dt / 0.18)
        rel = math.exp(-dt / 0.65)
        duck_100 = np.empty(n_bins, dtype=np.float32)
        cur_duck = gain_free
        for bi in range(n_bins):
            target = gain_under_voice if env_100[bi] > 0.5 else gain_free
            coef = atk if target < cur_duck else rel
            cur_duck = coef * cur_duck + (1.0 - coef) * target
            duck_100[bi] = cur_duck

        bin_x = (np.arange(n_bins, dtype=np.float64) + 0.5) * step
        duck = np.interp(np.arange(n_total, dtype=np.float64), bin_x, duck_100).astype(np.float32)

        fade_in_n = min(int(1.2 * SAMPLE_RATE), max(1, n_total // 4))
        fade_out_n = min(int(3.2 * SAMPLE_RATE), max(1, n_total // 3))
        fade_mul = np.ones(n_total, dtype=np.float32)
        fade_mul[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n, endpoint=False, dtype=np.float32)
        fade_mul[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n, endpoint=True, dtype=np.float32)

        m_idx = np.arange(n_total) % len(music_st)
        bgm_gain = duck * fade_mul
        ml = music_st[m_idx, 0] * bgm_gain
        mr = music_st[m_idx, 1] * bgm_gain
        l_mix = np.clip(base_mono + ml, -0.98, 0.98)
        r_mix = np.clip(base_mono + mr, -0.98, 0.98)
    else:
        l_mix = np.clip(base_mono, -0.98, 0.98)
        r_mix = l_mix

    stereo = np.stack([l_mix, r_mix], axis=1)
    pcm_bytes = np.round(stereo * 32767.0).astype("<i2").tobytes()

    pre_wav = os.path.join(audio_out_dir, "mixed_pre_norm.wav")
    with wave.open(pre_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    norm_wav = os.path.join(audio_out_dir, "master_audio.wav")
    base_af = "highpass=f=70,acompressor=threshold=-22dB:ratio=2.4:attack=15:release=220:makeup=2.5"
    p1 = subprocess.run(
        [
            "ffmpeg", "-y", "-i", pre_wav,
            "-af", f"{base_af},loudnorm=I={target_lufs:.1f}:LRA=6.5:TP=-2.2:print_format=json",
            "-f", "null", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    js_match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p1.stderr, re.S)
    applied_p2 = False
    if js_match:
        st = json.loads(js_match.group(0))
        keys = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
        if all(_is_finite_float(st.get(k)) for k in keys):
            p2_af = (
                f"{base_af},loudnorm=I={target_lufs:.1f}:LRA=6.5:TP=-2.2"
                f":measured_I={st['input_i']}:measured_LRA={st['input_lra']}"
                f":measured_TP={st['input_tp']}:measured_thresh={st['input_thresh']}"
                f":offset={st['target_offset']}:linear=true"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", pre_wav, "-af", p2_af,
                    "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", norm_wav,
                ],
                check=True,
                capture_output=True,
            )
            applied_p2 = True

    if not applied_p2:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", pre_wav, "-af", base_af,
                "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", norm_wav,
            ],
            check=True,
            capture_output=True,
        )

    recap_phrases = pages[-1].get("recap_phrases") or script.get("recap_phrases") or []
    recap_window: Optional[Dict[str, Any]] = None
    if recap_phrases:
        anchor_t = cues[-1]["t1"] if cues else max(0.0, total_dur - 3.0)
        t0 = round(min(anchor_t + 0.35, max(0.0, total_dur - 1.0)), 3)
        t1 = round(max(t0 + 0.8, total_dur - 0.15), 3)
        recap_window = {
            "t0": t0,
            "t1": t1,
            "title": str(script.get("recap_title") or "Key Phrases"),
            "phrases": list(recap_phrases),
        }

    return norm_wav, page_specs, cues, recap_window, total_dur, dissolve_sec


def render_static_ken_burns(
    img_path: str,
    out_mp4: str,
    dur: float,
    direction: str,
    apply_grade: bool = True,
) -> None:
    """Render smooth sub-pixel Ken Burns motion via 1.5x-oversampled cv2.warpAffine."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    ih, iw = img.shape[:2]
    target_ar = TARGET_W / float(TARGET_H)
    if iw / float(ih) > target_ar:
        nw = int(round(ih * target_ar))
        x0 = max(0, (iw - nw) // 2)
        cropped = img[:, x0 : x0 + nw]
    else:
        nh = int(round(iw / target_ar))
        y0 = max(0, (ih - nh) // 2)
        cropped = img[y0 : y0 + nh, :]

    hi = cv2.resize(cropped, (OVERSAMPLE_W, OVERSAMPLE_H), interpolation=cv2.INTER_LANCZOS4)
    delta_z = min(0.035, 0.0025 * dur)
    frames = max(2, int(round(dur * FPS)))
    vf = "eq=gamma=1.04:saturation=1.03,format=yuv420p" if apply_grade else "format=yuv420p"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{TARGET_W}x{TARGET_H}", "-r", str(FPS), "-i", "pipe:0",
        "-vf", vf,
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for i in range(frames):
            t = i / float(frames - 1)
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            if direction == "out":
                z = (1.0 + delta_z) - delta_z * eased
            else:
                z = 1.0 + delta_z * eased
            s = (TARGET_W / float(OVERSAMPLE_W)) * z
            tx = (TARGET_W - 1) * 0.5 - (OVERSAMPLE_W - 1) * 0.5 * s
            ty = (TARGET_H - 1) * 0.5 - (OVERSAMPLE_H - 1) * 0.5 * s
            mat = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)
            frame = cv2.warpAffine(
                hi,
                mat,
                (TARGET_W, TARGET_H),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass

    err_bytes = proc.stderr.read()
    proc.stderr.close()
    rc = proc.wait()
    if rc != 0:
        err_txt = err_bytes.decode("utf-8", errors="replace") if err_bytes else f"rc={rc}"
        raise RuntimeError(f"ffmpeg failed rendering {out_mp4}: {err_txt}")


def render_animated_page(
    clip_path: str,
    out_mp4: str,
    total_dur: float,
    work_dir: str,
    pid: str,
    anim_fps_mode: str = "1to1",
) -> None:
    """Conform animated clip and smoothly hold on its exact final frame."""
    slow_clip = os.path.join(work_dir, f"{pid}_slow.mp4")
    vf_parts = [
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={TARGET_W}:{TARGET_H}",
    ]
    if anim_fps_mode == "1to1":
        vf_parts.append(f"setpts=N/({FPS}*TB)")
    else:
        vf_parts.append(f"fps={FPS}")
    vf_parts.append("eq=gamma=1.04:saturation=1.03,format=yuv420p")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", clip_path, "-an",
            "-vf", ",".join(vf_parts),
            "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "15",
            slow_clip,
        ],
        check=True,
        capture_output=True,
    )
    slow_dur = probe_duration(slow_clip)
    if slow_dur >= total_dur - 0.15:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", slow_clip, "-t", f"{total_dur:.3f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "16", out_mp4,
            ],
            check=True,
            capture_output=True,
        )
        return

    last_png = os.path.join(work_dir, f"{pid}_lastframe.png")
    cap = cv2.VideoCapture(slow_clip)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_f > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_f - 10)
    last_frame = None
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        last_frame = frm
    cap.release()
    if last_frame is None:
        raise RuntimeError(f"Failed to extract final frame from {slow_clip}")
    cv2.imwrite(last_png, last_frame)

    blend = min(0.45, max(0.10, slow_dur * 0.25))
    hold_dur = max(0.25, (total_dur - slow_dur) + blend)
    hold_mp4 = os.path.join(work_dir, f"{pid}_hold.mp4")
    render_static_ken_burns(last_png, hold_mp4, hold_dur, "in", apply_grade=False)

    xf_off = max(0.05, slow_dur - blend)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", slow_clip, "-i", hold_mp4,
            "-filter_complex", f"[0:v][1:v]xfade=transition=fade:duration={blend:.3f}:offset={xf_off:.3f},format=yuv420p[v]",
            "-map", "[v]", "-t", f"{total_dur:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", out_mp4,
        ],
        check=True,
        capture_output=True,
    )


def wrap_en_lines(draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        cand = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), cand, font=font)
        if (bbox[2] - bbox[0]) <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return lines


def wrap_zh_lines(draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int) -> List[str]:
    clean = text.strip()
    if not clean:
        return []
    bbox = draw.textbbox((0, 0), clean, font=font)
    if (bbox[2] - bbox[0]) <= max_w:
        return [clean]
    mid = len(clean) // 2
    best_idx = mid
    best_dist = 999
    for idx, ch in enumerate(clean):
        if ch in "，。！？；：…" and 3 <= idx <= len(clean) - 3:
            if abs(idx + 1 - mid) < best_dist:
                best_dist = abs(idx + 1 - mid)
                best_idx = idx + 1
    if best_dist < 999:
        return [clean[:best_idx].strip(), clean[best_idx:].strip()]
    return [clean[:mid], clean[mid:]]


def render_story_cue_card(
    out_path: str,
    speaker: str,
    en_text: str,
    zh_text: str,
    speakers_cfg: Dict[str, Any],
    en_font_path: Optional[str] = None,
    zh_font_path: Optional[str] = None,
) -> Tuple[int, int]:
    en_font = load_font(en_font_path, EN_FONT_CANDIDATES, 42)
    zh_font = load_font(zh_font_path, ZH_FONT_CANDIDATES, 31)
    tag_font = load_font(zh_font_path, ZH_FONT_CANDIDATES, 22)
    sp = resolve_speaker_style(speaker, speakers_cfg)
    tag_text = sp.get("tag")

    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(dummy)
    en_lines = wrap_en_lines(d0, en_text or "", en_font, 840)
    zh_lines = wrap_zh_lines(d0, zh_text or "", zh_font, 840)
    if not en_lines and not zh_lines:
        en_lines = [" "]

    en_line_h = 50
    zh_line_h = 40
    max_text_w = 0
    for ln in en_lines:
        b = d0.textbbox((0, 0), ln, font=en_font)
        max_text_w = max(max_text_w, b[2] - b[0])
    for ln in zh_lines:
        b = d0.textbbox((0, 0), ln, font=zh_font)
        max_text_w = max(max_text_w, b[2] - b[0])

    pad_x = 40
    pad_y = 22 if tag_text else 20
    gap = 6 if (en_lines and zh_lines) else 0
    box_w = min(980, max(380, max_text_w + pad_x * 2))
    box_h = pad_y * 2 + len(en_lines) * en_line_h + len(zh_lines) * zh_line_h + gap + 4

    top_offset = 24 if tag_text else 4
    img = Image.new("RGBA", (box_w + 8, top_offset + box_h + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [4, top_offset, box_w + 4, top_offset + box_h],
        radius=28,
        fill=(26, 22, 34, 214),
        outline=sp["outline"],
        width=3,
    )

    cur_y = top_offset + pad_y
    for ln in en_lines:
        b = d.textbbox((0, 0), ln, font=en_font)
        tw = b[2] - b[0]
        x = 4 + (box_w - tw) // 2 - b[0]
        y = cur_y - b[1] + 4
        d.text((x + 2, y + 2), ln, font=en_font, fill=(0, 0, 0, 180))
        d.text((x, y), ln, font=en_font, fill=sp["fill"])
        cur_y += en_line_h

    if en_lines and zh_lines:
        cur_y += gap
    for ln in zh_lines:
        b = d.textbbox((0, 0), ln, font=zh_font)
        tw = b[2] - b[0]
        x = 4 + (box_w - tw) // 2 - b[0]
        y = cur_y - b[1] + 2
        d.text((x + 1, y + 1), ln, font=zh_font, fill=(0, 0, 0, 160))
        d.text((x, y), ln, font=zh_font, fill=(252, 248, 240, 255))
        cur_y += zh_line_h

    if tag_text:
        tb = d.textbbox((0, 0), str(tag_text), font=tag_font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        pill_w = tw + 28
        pill_h = 36
        px0 = 32
        py0 = 2
        d.rounded_rectangle(
            [px0, py0, px0 + pill_w, py0 + pill_h],
            radius=18,
            fill=sp["tag_bg"],
            outline=(26, 22, 34, 240),
            width=2,
        )
        tx = px0 + (pill_w - tw) // 2 - tb[0]
        ty = py0 + (pill_h - th) // 2 - tb[1] - 1
        d.text((tx, ty), str(tag_text), font=tag_font, fill=sp["tag_fg"])

    img.save(out_path, format="PNG")
    return img.size


def render_recap_card(
    out_path: str,
    title: str,
    phrases: List[str],
    en_font_path: Optional[str] = None,
    zh_font_path: Optional[str] = None,
) -> Tuple[int, int]:
    hdr_font = load_font(zh_font_path, ZH_FONT_CANDIDATES, 32)
    ph_font = load_font(en_font_path, EN_FONT_CANDIDATES, 40)
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(dummy)
    body = "   /   ".join(phrases)
    b1 = d0.textbbox((0, 0), title, font=hdr_font)
    b2 = d0.textbbox((0, 0), body, font=ph_font)
    if (b2[2] - b2[0]) > 880:
        ph_font = load_font(en_font_path, EN_FONT_CANDIDATES, 32)
        b2 = d0.textbbox((0, 0), body, font=ph_font)

    w = min(980, max(360, max(b1[2] - b1[0], b2[2] - b2[0]) + 84))
    h = 152
    img = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [4, 4, w + 4, h + 4],
        radius=30,
        fill=(26, 22, 34, 215),
        outline=(255, 220, 120, 235),
        width=3,
    )
    hw = b1[2] - b1[0]
    hx = 4 + (w - hw) // 2 - b1[0]
    d.text((hx, 22 - b1[1]), title, font=hdr_font, fill=(255, 238, 185, 255))

    bw = b2[2] - b2[0]
    bx = 4 + (w - bw) // 2 - b2[0]
    d.text((bx + 2, 78 - b2[1]), body, font=ph_font, fill=(0, 0, 0, 180))
    d.text((bx, 76 - b2[1]), body, font=ph_font, fill=(255, 215, 90, 255))
    img.save(out_path, format="PNG")
    return img.size


def render_badge(out_path: str, label: str, zh_font_path: Optional[str] = None) -> None:
    font = load_font(zh_font_path, ZH_FONT_CANDIDATES, 28)
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(dummy)
    b = d0.textbbox((0, 0), label, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    pad_x, pad_y = 22, 12
    w, h = tw + pad_x * 2, th + pad_y * 2
    img = Image.new("RGBA", (w + 6, h + 6), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [3, 3, w + 3, h + 3],
        radius=20,
        fill=(26, 22, 34, 180),
        outline=(255, 214, 102, 210),
        width=2,
    )
    d.text((3 + pad_x - b[0], 3 + pad_y - b[1]), label, font=font, fill=(255, 234, 170, 248))
    img.save(out_path, format="PNG")


def burn_overlay_version(
    master: str,
    out_mp4: str,
    badge_path: str,
    card_infos: List[Dict[str, Any]],
    recap_info: Optional[Dict[str, Any]],
    bottom_y: int,
    script_path: str,
) -> None:
    """Burn subtitle pills, optional badge, and optional recap card via -filter_complex_script."""
    overlay_items: List[Tuple[str, str]] = []
    cmd = ["ffmpeg", "-y", "-i", master]
    lines: List[str] = []
    prev = "[0:v]"
    idx_counter = 1

    if badge_path:
        cmd.extend(["-i", badge_path])
        overlay_items.append(("badge", "x=W-w-36:y=96"))

    for ci in card_infos:
        cmd.extend(["-i", ci["png"]])
        y_pos = bottom_y - ci["h"]
        expr = f"between(t,{ci['t0']:.3f},{ci['t1']:.3f})"
        overlay_items.append(("card", f"x=(W-w)/2:y={y_pos}:enable='{expr}'"))

    if recap_info is not None:
        cmd.extend(["-i", recap_info["png"]])
        recap_y = bottom_y - recap_info["h"]
        recap_expr = f"between(t,{recap_info['t0']:.3f},{recap_info['t1']:.3f})"
        overlay_items.append(("recap", f"x=(W-w)/2:y={recap_y}:enable='{recap_expr}'"))

    if not overlay_items:
        subprocess.run(
            ["ffmpeg", "-y", "-i", master, "-c", "copy", "-movflags", "+faststart", out_mp4],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    total_ov = len(overlay_items)
    for k, (_, opt_str) in enumerate(overlay_items):
        out_lbl = "[vout]" if k == total_ov - 1 else f"[v{idx_counter}]"
        lines.append(f"{prev}[{idx_counter}:v]overlay={opt_str}{out_lbl}")
        prev = out_lbl
        idx_counter += 1

    flt_str = ";".join(lines)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(";\n".join(lines))

    if len(flt_str) <= 65536:
        filter_args = ["-filter_complex", flt_str]
    else:
        help_txt = subprocess.run(["ffmpeg", "-h", "long"], capture_output=True, text=True).stdout
        if "-filter_complex_script" in help_txt:
            filter_args = ["-filter_complex_script", script_path]
        else:
            filter_args = ["-/filter_complex", script_path]

    cmd.extend(filter_args + [
        "-map", "[vout]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "slow", "-b:v", "8.5M", "-maxrate", "12M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "copy", "-movflags", "+faststart",
        out_mp4,
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print("ffmpeg burn failed:\n" + (exc.stderr or "")[-2000:], file=sys.stderr)
        raise


def parse_presets(presets_raw: str) -> List[str]:
    allowed = {"clean", "shorts", "douyin", "xiaohongshu"}
    items = [p.strip().lower() for p in presets_raw.split(",") if p.strip()]
    if not items:
        return ["shorts"]
    for it in items:
        if it not in allowed:
            raise ValueError(f"Unsupported preset '{it}'. Choose from: {sorted(allowed)}")
    return list(dict.fromkeys(items))


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble 1080x1920 @ 30fps storybook MP4.")
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing script.json and pages/")
    parser.add_argument("--prefix", default="ep01", help="Output filename prefix (default: ep01)")
    parser.add_argument("--badge", default="", help="Optional top-right badge label text")
    parser.add_argument("--bgm", default=None, help="Optional BGM path (or 'none' to disable)")
    parser.add_argument("--target-lufs", type=float, default=-15.0, help="Target integrated loudness (default: -15.0)")
    parser.add_argument(
        "--presets",
        default="shorts",
        help="Comma-separated outputs: clean,shorts,douyin,xiaohongshu (default: shorts)",
    )
    parser.add_argument("--subtitle-bottom-y", type=int, default=None, help="Override subtitle pill bottom Y edge")
    parser.add_argument("--anim-fps-mode", choices=["1to1", "resample"], default="1to1", help="Animated clip frame mapping")
    parser.add_argument("--en-font", default=None, help="Optional custom TTF/TTC font for English text")
    parser.add_argument("--zh-font", default=None, help="Optional custom TTF/TTC font for CJK/badge text")
    parser.add_argument("--rebuild-segments", action="store_true", help="Ignore cached per-page segments")
    parser.add_argument("--reoverlay-only", action="store_true", help="Reuse clean_master.mp4 and re-burn overlays")
    args = parser.parse_args()

    presets = parse_presets(args.presets)
    ep_dir = os.path.abspath(args.episode_dir)
    pages_dir = os.path.join(ep_dir, "pages")
    clips_dir = os.path.join(ep_dir, "clips")
    work_dir = os.path.join(ep_dir, "_build_tmp")
    cards_dir = os.path.join(work_dir, "cards")
    os.makedirs(cards_dir, exist_ok=True)

    clean_master = os.path.join(ep_dir, f"{args.prefix}_clean_master.mp4")
    cues_json_path = os.path.join(ep_dir, "synced_cues.json")

    with open(os.path.join(ep_dir, "script.json"), "r", encoding="utf-8") as f:
        script = json.load(f)
    speakers_cfg = script.get("speakers", {}) if isinstance(script.get("speakers"), dict) else {}

    if args.reoverlay_only:
        if not os.path.exists(clean_master) or not os.path.exists(cues_json_path):
            raise FileNotFoundError("--reoverlay-only requires existing clean_master MP4 and synced_cues.json")
        print(f"Reusing clean master & cues: {clean_master}")
        with open(cues_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cues = meta.get("cues", [])
        recap_win = meta.get("recap")
    else:
        print("1) Building audio mix & exact subtitle timeline...")
        master_wav, page_specs, cues, recap_win, total_dur, dissolve_sec = build_audio_and_timeline(
            ep_dir,
            work_dir,
            script,
            bgm_arg=args.bgm,
            target_lufs=args.target_lufs,
        )
        recap_msg = f"{recap_win['t0']}..{recap_win['t1']}s" if recap_win else "none"
        print(f"   Master audio ready: total_dur={total_dur:.2f}s, cues={len(cues)}, recap={recap_msg}")

        cues_out = {
            "total_duration_sec": total_dur,
            "pages": page_specs,
            "cues": [{k: v for k, v in c.items() if k not in ("samples", "sfx")} for c in cues],
            "recap": recap_win,
        }
        with open(cues_json_path, "w", encoding="utf-8") as f:
            json.dump(cues_out, f, ensure_ascii=False, indent=2)

        print("2) Rendering per-page 1080x1920@30fps segments...")
        seg_paths: List[str] = []
        for p in page_specs:
            pid = p["page_id"]
            out_seg = os.path.join(work_dir, f"{pid}_seg.mp4")
            if not args.rebuild_segments and os.path.exists(out_seg) and abs(probe_duration(out_seg) - p["duration"]) < 0.25:
                print(f"   [CACHED] {pid}: {p['duration']:.2f}s ({os.path.basename(out_seg)})")
                seg_paths.append(out_seg)
                continue
            clip_cand = os.path.join(clips_dir, f"{pid}_anim.mp4")
            if p["animated"] and os.path.exists(clip_cand):
                print(f"   [ANIM  ] {pid}: {p['duration']:.2f}s (from {os.path.basename(clip_cand)} + hold)")
                render_animated_page(clip_cand, out_seg, p["duration"], work_dir, pid, anim_fps_mode=args.anim_fps_mode)
            else:
                img_p = find_page_image(pages_dir, pid)
                print(f"   [KEN-B ] {pid}: {p['duration']:.2f}s ({p['ken_burns']}) from {os.path.basename(img_p)}")
                render_static_ken_burns(img_p, out_seg, p["duration"], p["ken_burns"])
            seg_paths.append(out_seg)

        print(f"3) Joining {len(seg_paths)} page(s) and muxing clean_master.mp4...")
        if len(seg_paths) == 1:
            cmd = [
                "ffmpeg", "-y", "-i", seg_paths[0], "-i", master_wav,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "slow", "-b:v", "8.5M", "-maxrate", "12M", "-bufsize", "16M",
                "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
                "-shortest", "-movflags", "+faststart",
                clean_master,
            ]
        else:
            cmd = ["ffmpeg", "-y"]
            for sp in seg_paths:
                cmd.extend(["-i", sp])
            cmd.extend(["-i", master_wav])

            flt_lines = []
            offset_acc = page_specs[0]["duration"] - dissolve_sec
            prev_lbl = "[0:v]"
            for i in range(1, len(seg_paths)):
                out_lbl = f"[xf{i}]" if i < len(seg_paths) - 1 else "[vout]"
                flt_lines.append(
                    f"{prev_lbl}[{i}:v]xfade=transition=fade:duration={dissolve_sec:.3f}:offset={offset_acc:.3f}{out_lbl}"
                )
                prev_lbl = out_lbl
                if i < len(seg_paths) - 1:
                    offset_acc += page_specs[i]["duration"] - dissolve_sec

            audio_idx = len(seg_paths)
            cmd.extend([
                "-filter_complex", ";".join(flt_lines),
                "-map", "[vout]", "-map", f"{audio_idx}:a",
                "-c:v", "libx264", "-preset", "slow", "-b:v", "8.5M", "-maxrate", "12M", "-bufsize", "16M",
                "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
                "-shortest", "-movflags", "+faststart",
                clean_master,
            ])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print("ffmpeg master mux failed:\n" + (exc.stderr or "")[-2000:], file=sys.stderr)
            raise
        print(f"   Saved {clean_master}")

    render_presets = [p for p in presets if p != "clean"]
    if not render_presets and args.subtitle_bottom_y is not None:
        render_presets = ["custom"]

    if not render_presets:
        print("4) Preset is 'clean' only; skipping overlay burn.")
        return 0

    print("4) Rendering subtitle cards, optional overlays, and preset outputs...")
    badge_png = ""
    if args.badge.strip():
        badge_png = os.path.join(cards_dir, "badge.png")
        render_badge(badge_png, args.badge.strip(), zh_font_path=args.zh_font)

    card_infos: List[Dict[str, Any]] = []
    for c in cues:
        p = os.path.join(cards_dir, f"cue_{c['line_id']}.png")
        _, h = render_story_cue_card(
            p,
            c["speaker"],
            c.get("en", ""),
            c.get("zh", ""),
            speakers_cfg,
            en_font_path=args.en_font,
            zh_font_path=args.zh_font,
        )
        card_infos.append({"png": p, "t0": c["t0"], "t1": c["t1"], "h": h})

    recap_info: Optional[Dict[str, Any]] = None
    if recap_win and recap_win.get("phrases"):
        recap_png = os.path.join(cards_dir, "recap.png")
        _, rh = render_recap_card(
            recap_png,
            recap_win.get("title") or "Key Phrases",
            recap_win["phrases"],
            en_font_path=args.en_font,
            zh_font_path=args.zh_font,
        )
        recap_info = {"png": recap_png, "t0": recap_win["t0"], "t1": recap_win["t1"], "h": rh}

    suffix_map = {
        "shorts": "shorts_ready",
        "douyin": "douyin_ready",
        "xiaohongshu": "xhs_ready",
        "custom": "subtitled",
    }
    for preset in render_presets:
        bottom_y = args.subtitle_bottom_y if args.subtitle_bottom_y is not None else PRESET_BOTTOM_Y[preset]
        out_mp4 = os.path.join(ep_dir, f"{args.prefix}_{suffix_map[preset]}.mp4")
        burn_script = os.path.join(work_dir, f"{preset}_burn.txt")
        burn_overlay_version(clean_master, out_mp4, badge_png, card_infos, recap_info, bottom_y, burn_script)
        print(f"   Saved {out_mp4} (bottom_y={bottom_y})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

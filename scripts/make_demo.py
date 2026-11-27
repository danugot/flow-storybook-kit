#!/usr/bin/env python3
"""Zero-API-key, zero-credit end-to-end demo generator for flow-storybook-kit.

Creates a synthetic 3-page vertical storybook episode with:
- 3 richly textured 1536x2752 (2K-scale) illustrated PNG pages
- 1 real 24fps 4-second animated clip (`clips/p02_anim.mp4`) for testing clip + final-frame hold
- 4 pleasant harmonic voice-like WAV files across MIXED sample rates & bit depths
  (24kHz 16-bit, 44.1kHz 16-bit, 48kHz 16-bit, 48kHz 24-bit)
- 1 soft stereo 44.1kHz BGM bed + 1 chime foley SFX
- A generic `script.json` with Narrator / Fox / Owl characters
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import wave
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

WIDTH = 1536
HEIGHT = 2752


def _write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clipped = np.clip(samples, -0.99, 0.99)
    if sampwidth == 2:
        raw = np.round(clipped * 32767.0).astype("<i2").tobytes()
    elif sampwidth == 3:
        scaled = np.round(clipped * 8388607.0).astype("<i4").reshape(-1)
        b = scaled.view(np.uint8).reshape(-1, 4)[:, :3]
        raw = b.tobytes()
    else:
        raise ValueError(f"Unsupported sampwidth: {sampwidth}")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def _synth_phrase_audio(freqs: Sequence[float], duration: float, sample_rate: int, amp: float = 0.28) -> np.ndarray:
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    out = np.zeros(n, dtype=np.float64)
    seg_len = max(1, n // len(freqs))
    for i, f0 in enumerate(freqs):
        s0 = i * seg_len
        s1 = n if i == len(freqs) - 1 else (i + 1) * seg_len
        lt = t[s0:s1] - t[s0]
        tone = (
            0.65 * np.sin(2.0 * math.pi * f0 * lt)
            + 0.25 * np.sin(2.0 * math.pi * (f0 * 2.0) * lt)
            + 0.10 * np.sin(2.0 * math.pi * (f0 * 3.0) * lt)
        )
        env = np.minimum(1.0, lt / 0.035) * np.minimum(1.0, (lt[-1] - lt + 1e-4) / 0.08)
        out[s0:s1] = tone * env * amp
    return out


def _paint_storybook_page(path: str, variant: int, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Bright twilight/pastel sky gradient (mean luma ~110-135 for clean mobile feed readability)
    y_grad = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    if variant == 1:
        top = np.array([78, 108, 168], dtype=np.float32)
        bot = np.array([196, 178, 152], dtype=np.float32)
    elif variant == 2:
        top = np.array([68, 118, 166], dtype=np.float32)
        bot = np.array([186, 198, 162], dtype=np.float32)
    else:
        top = np.array([82, 104, 158], dtype=np.float32)
        bot = np.array([178, 168, 198], dtype=np.float32)

    canvas = (top[None, None, :] * (1.0 - y_grad[:, :, None]) + bot[None, None, :] * y_grad[:, :, None])
    # Organic storybook paper fiber texture so H.264 ABR bitrate matches real illustrated pages
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 9.5, size=(HEIGHT, WIDTH, 3)).astype(np.float32)
    textured = np.clip(canvas + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(textured, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Moon and soft glow
    moon_x = 1140 if variant != 2 else 420
    moon_y = 420 if variant != 3 else 360
    draw.ellipse([moon_x - 170, moon_y - 170, moon_x + 170, moon_y + 170], fill=(248, 236, 185))
    draw.ellipse([moon_x - 125, moon_y - 125, moon_x + 125, moon_y + 125], fill=(255, 248, 218))

    # Rolling hills
    draw.ellipse([-300, 1780, 1300, 2950], fill=(92, 134, 104))
    draw.ellipse([360, 1920, 1950, 3050], fill=(76, 116, 92))

    # Cozy cabin / lantern motif
    cx = 420 + variant * 160
    cy = 1520
    draw.rectangle([cx - 140, cy, cx + 140, cy + 260], fill=(168, 124, 92))
    draw.polygon([(cx - 180, cy + 10), (cx, cy - 130), (cx + 180, cy + 10)], fill=(138, 86, 68))
    draw.rectangle([cx - 48, cy + 70, cx + 48, cy + 175], fill=(255, 214, 120))

    # Character badges/silhouettes (Fox + Owl shapes)
    fx = 680 if variant == 1 else (760 if variant == 2 else 620)
    fy = 1960
    draw.ellipse([fx - 95, fy - 95, fx + 95, fy + 95], fill=(238, 146, 82))
    draw.polygon([(fx - 80, fy - 45), (fx - 45, fy - 145), (fx - 15, fy - 75)], fill=(238, 146, 82))
    draw.polygon([(fx + 15, fy - 75), (fx + 45, fy - 145), (fx + 80, fy - 45)], fill=(238, 146, 82))
    draw.ellipse([fx - 42, fy - 25, fx - 18, fy - 1], fill=(42, 34, 30))
    draw.ellipse([fx + 18, fy - 25, fx + 42, fy - 1], fill=(42, 34, 30))

    # Glowing lantern orb
    lx = fx + 155
    ly = fy + 25
    draw.ellipse([lx - 55, ly - 55, lx + 55, ly + 55], fill=(255, 224, 126))
    draw.ellipse([lx - 32, ly - 32, lx + 32, ly + 32], fill=(255, 245, 200))

    # Re-apply subtle texture over flat vector shapes for realistic compression density
    arr = np.asarray(img, dtype=np.float32)
    fine_grain = rng.normal(0.0, 5.5, size=(HEIGHT, WIDTH, 3)).astype(np.float32)
    final_img = Image.fromarray(np.clip(arr + fine_grain, 0, 255).astype(np.uint8), mode="RGB")
    final_img.save(path, format="PNG")


def _render_demo_anim_clip(page_png: str, out_mp4: str, dur: float = 4.0, fps: int = 24) -> None:
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    src = cv2.imread(page_png, cv2.IMREAD_COLOR)
    assert src is not None
    w, h = 720, 1280
    base = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    frames = int(round(dur * fps))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    fireflies: Tuple[Tuple[int, int, float], ...] = (
        (180, 920, 0.0),
        (310, 860, 1.1),
        (510, 940, 2.3),
        (430, 810, 0.7),
        (260, 760, 1.9),
    )
    for i in range(frames):
        t = i / float(fps)
        frame = base.copy()
        for bx, by, phase in fireflies:
            cx = int(round(bx + 18.0 * math.sin(2.0 * t + phase)))
            cy = int(round(by - 28.0 * t + 10.0 * math.cos(2.5 * t + phase)))
            cv2.circle(frame, (cx, cy), 10, (130, 225, 255), -1)
            cv2.circle(frame, (cx, cy), 5, (210, 250, 255), -1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Failed rendering demo animated clip: rc={rc}")


def build_demo_episode(out_dir: str) -> str:
    ep_dir = os.path.abspath(out_dir)
    pages_dir = os.path.join(ep_dir, "pages")
    clips_dir = os.path.join(ep_dir, "clips")
    lines_dir = os.path.join(ep_dir, "audio", "lines")
    foley_dir = os.path.join(ep_dir, "audio", "foley")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(foley_dir, exist_ok=True)

    _paint_storybook_page(os.path.join(pages_dir, "p01.png"), variant=1, seed=101)
    _paint_storybook_page(os.path.join(pages_dir, "p02.png"), variant=2, seed=202)
    _paint_storybook_page(os.path.join(pages_dir, "p03.png"), variant=3, seed=303)
    _render_demo_anim_clip(os.path.join(pages_dir, "p02.png"), os.path.join(clips_dir, "p02_anim.mp4"), dur=4.0, fps=24)

    # Mixed sample rates & bit depths prove universal ffmpeg audio ingestion
    _write_pcm_wav(
        os.path.join(lines_dir, "line_01.wav"),
        _synth_phrase_audio([392.0, 440.0, 523.25], 2.4, 24000),
        sample_rate=24000,
        sampwidth=2,
    )
    _write_pcm_wav(
        os.path.join(lines_dir, "line_02.wav"),
        _synth_phrase_audio([523.25, 587.33, 659.25], 2.2, 44100),
        sample_rate=44100,
        sampwidth=2,
    )
    _write_pcm_wav(
        os.path.join(lines_dir, "line_03.wav"),
        _synth_phrase_audio([329.63, 392.0, 440.0], 2.5, 48000),
        sample_rate=48000,
        sampwidth=2,
    )
    _write_pcm_wav(
        os.path.join(lines_dir, "line_04.wav"),
        _synth_phrase_audio([440.0, 392.0, 329.63], 2.6, 48000),
        sample_rate=48000,
        sampwidth=3,  # 24-bit PCM
    )

    # Soft stereo BGM loop (44.1kHz) + foley chime (48kHz)
    bgm_dur = 8.0
    bgm_sr = 44100
    t_b = np.arange(int(bgm_dur * bgm_sr), dtype=np.float64) / float(bgm_sr)
    l_bgm = 0.14 * np.sin(2.0 * math.pi * 261.63 * t_b) + 0.08 * np.sin(2.0 * math.pi * 392.0 * t_b)
    r_bgm = 0.14 * np.sin(2.0 * math.pi * 261.63 * t_b) + 0.08 * np.sin(2.0 * math.pi * 329.63 * t_b)
    _write_pcm_wav(os.path.join(ep_dir, "audio", "bgm.wav"), np.stack([l_bgm, r_bgm], axis=1), bgm_sr, channels=2)

    t_c = np.arange(int(0.9 * 48000), dtype=np.float64) / 48000.0
    chime = 0.35 * np.sin(2.0 * math.pi * 880.0 * t_c) * np.exp(-t_c * 4.5)
    _write_pcm_wav(os.path.join(foley_dir, "chime.wav"), chime, 48000, channels=1)

    script = {
        "episode_id": os.path.basename(ep_dir),
        "title": "The Little Lantern Walk",
        "recap_title": "Tonight's 3 Phrases",
        "bgm": "audio/bgm.wav",
        "voices": {
            "Narrator": {"voice": "Vindemiatrix", "style": "Warm bedtime narrator"},
            "Fox": {"voice": "Achird", "style": "Bright curious young fox"},
            "Owl": {"voice": "Enceladus", "style": "Calm wise owl"},
        },
        "speakers": {
            "Narrator": {"fill": [255, 224, 130, 255], "outline": [255, 214, 102, 225], "tag": None},
            "Fox": {
                "fill": [255, 196, 112, 255],
                "outline": [255, 178, 92, 225],
                "tag": "Little Fox",
                "tag_bg": [255, 186, 98, 248],
                "tag_fg": [42, 20, 8, 255],
            },
            "Owl": {
                "fill": [150, 222, 255, 255],
                "outline": [130, 206, 255, 225],
                "tag": "Grandpa Owl",
                "tag_bg": [130, 206, 255, 248],
                "tag_fg": [18, 28, 42, 255],
            },
        },
        "pages": [
            {
                "page_id": "p01",
                "animated": False,
                "ken_burns": "in",
                "image_prompt": "Vertical 9:16 storybook forest cabin at moonrise with a glowing lantern.",
                "lines": [
                    {
                        "line_id": "01",
                        "speaker": "Narrator",
                        "en": "Click! The paper lantern glows softly under the moon.",
                        "zh": "咔哒！小纸灯笼在月光下轻轻亮了起来。",
                        "sfx": {"file": "audio/foley/chime.wav", "offset_sec": -0.1, "gain": 0.25},
                    },
                    {
                        "line_id": "02",
                        "speaker": "Fox",
                        "en": "Look, Grandpa Owl! My lantern is ready!",
                        "zh": "快看，猫头鹰爷爷！我的灯笼准备好啦！",
                    },
                ],
            },
            {
                "page_id": "p02",
                "animated": True,
                "clip_duration_target": 5,
                "ip_sensitive_for_veo": False,
                "silent_guard": True,
                "ken_burns": "out",
                "motion_prompt": "Fireflies drift gently upward around the little fox holding a warm lantern.",
                "lines": [
                    {
                        "line_id": "03",
                        "speaker": "Owl",
                        "en": "Hold it steady, little fox. Follow the quiet trail.",
                        "zh": "拿稳了哦，小狐狸。顺着安静的小路走吧。",
                    }
                ],
            },
            {
                "page_id": "p03",
                "animated": False,
                "ken_burns": "in",
                "hold_tail_sec": 3.2,
                "recap_phrases": ["My lantern is ready!", "Hold it steady.", "Goodnight, forest."],
                "image_prompt": "Vertical 9:16 cozy twilight hillside under starry skies.",
                "lines": [
                    {
                        "line_id": "04",
                        "speaker": "Narrator",
                        "en": "Safe and warm beneath the stars. Goodnight, forest.",
                        "zh": "在星空下安心又温暖。晚安，小森林。",
                    }
                ],
            },
        ],
    }

    with open(os.path.join(ep_dir, "script.json"), "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    return ep_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a zero-cost demo episode to verify the pipeline.")
    parser.add_argument("--out-dir", default="demo_episode", help="Output directory (default: demo_episode)")
    args = parser.parse_args()
    ep_dir = build_demo_episode(args.out_dir)
    print(f"Demo episode created at: {ep_dir}")
    print("Next steps:")
    print(f"  python scripts/assemble_storybook.py --episode-dir {args.out_dir} --prefix demo --presets shorts,xiaohongshu,douyin")
    print(f"  python scripts/audit_feed_delivery.py {args.out_dir}/demo_shorts_ready.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

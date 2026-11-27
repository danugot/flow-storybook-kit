#!/usr/bin/env python3
"""Comprehensive test suite for flow-storybook-kit."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import assemble_storybook as asm  # noqa: E402
import flow_route_example as fre  # noqa: E402
import gen_tts_lines as gtts  # noqa: E402
import make_demo as mdemo  # noqa: E402


def _run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)] + args,
        capture_output=True,
        text=True,
    )


def test_end_to_end_demo_assemble_audit_and_reoverlay(tmp_path: Path) -> None:
    ep_dir = tmp_path / "demo_ep"
    mdemo.build_demo_episode(str(ep_dir))

    res = _run_script(
        "assemble_storybook.py",
        [
            "--episode-dir", str(ep_dir),
            "--prefix", "demo",
            "--presets", "shorts,xiaohongshu,douyin",
            "--badge", "Demo",
        ],
    )
    assert res.returncode == 0, res.stderr

    clean_mp4 = ep_dir / "demo_clean_master.mp4"
    shorts_mp4 = ep_dir / "demo_shorts_ready.mp4"
    xhs_mp4 = ep_dir / "demo_xhs_ready.mp4"
    dy_mp4 = ep_dir / "demo_douyin_ready.mp4"
    cues_json = ep_dir / "synced_cues.json"
    for p in (clean_mp4, shorts_mp4, xhs_mp4, dy_mp4):
        assert p.exists() and p.stat().st_size > 10_000
    assert cues_json.exists() and cues_json.stat().st_size > 200

    aud = _run_script("audit_feed_delivery.py", [str(shorts_mp4), "--json"])
    assert aud.returncode == 0, aud.stdout + aud.stderr
    report = json.loads(aud.stdout)
    assert report["fail_count"] == 0, report
    assert report["probe"]["width"] == 1080
    assert report["probe"]["height"] == 1920
    assert report["probe"]["fps"] == 30.0
    assert -16.5 <= report["audio"]["integrated_lufs"] <= -13.0
    assert report["luma"]["yavg_mean"] >= 60.0

    # Verify --reoverlay-only regenerates preset output without touching clean_master
    master_mtime_before = clean_mp4.stat().st_mtime_ns
    shorts_mp4.unlink()
    reov = _run_script(
        "assemble_storybook.py",
        ["--episode-dir", str(ep_dir), "--prefix", "demo", "--presets", "shorts", "--reoverlay-only"],
    )
    assert reov.returncode == 0, reov.stderr
    assert shorts_mp4.exists()
    assert clean_mp4.stat().st_mtime_ns == master_mtime_before


def test_single_page_no_bgm_english_only_no_recap(tmp_path: Path) -> None:
    ep_dir = tmp_path / "single_ep"
    (ep_dir / "pages").mkdir(parents=True)
    (ep_dir / "audio" / "lines").mkdir(parents=True)

    mdemo._paint_storybook_page(str(ep_dir / "pages" / "p01.png"), variant=1, seed=7)
    mdemo._write_pcm_wav(
        str(ep_dir / "audio" / "lines" / "line_01.wav"),
        mdemo._synth_phrase_audio([440.0, 523.25], 1.8, 24000),
        sample_rate=24000,
    )
    script = {
        "episode_id": "single_ep",
        "pages": [
            {
                "page_id": "p01",
                "animated": False,
                "ken_burns": "in",
                "lines": [
                    {
                        "line_id": "01",
                        "speaker": "Bear",  # arbitrary speaker triggers fallback palette
                        "en": "Hello starry sky.",
                        "zh": "",
                    }
                ],
            }
        ],
    }
    (ep_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")

    res = _run_script(
        "assemble_storybook.py",
        ["--episode-dir", str(ep_dir), "--prefix", "one", "--presets", "clean,shorts"],
    )
    assert res.returncode == 0, res.stderr
    assert (ep_dir / "one_clean_master.mp4").exists()
    assert (ep_dir / "one_shorts_ready.mp4").exists()


def test_empty_lines_silent_episode_no_crash(tmp_path: Path) -> None:
    ep_dir = tmp_path / "silent_ep"
    (ep_dir / "pages").mkdir(parents=True)
    (ep_dir / "audio" / "lines").mkdir(parents=True)

    img = Image.new("RGB", (1080, 1920), (140, 150, 175))
    img.save(ep_dir / "pages" / "p01.png")
    img.save(ep_dir / "pages" / "p02.png")

    script = {
        "episode_id": "silent_ep",
        "pages": [
            {"page_id": "p01", "duration_sec": 1.6, "lines": []},
            {"page_id": "p02", "duration_sec": 1.6, "lines": []},
        ],
    }
    (ep_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")

    res = _run_script(
        "assemble_storybook.py",
        ["--episode-dir", str(ep_dir), "--prefix", "sil", "--presets", "shorts"],
    )
    assert res.returncode == 0, res.stderr
    assert (ep_dir / "sil_shorts_ready.mp4").exists()


def test_mixed_sample_rates_and_mp3_ingestion(tmp_path: Path) -> None:
    ep_dir = tmp_path / "mixed_audio_ep"
    (ep_dir / "pages").mkdir(parents=True)
    lines_dir = ep_dir / "audio" / "lines"
    lines_dir.mkdir(parents=True)

    Image.new("RGB", (1080, 1920), (130, 145, 165)).save(ep_dir / "pages" / "p01.png")
    wav_16k = lines_dir / "line_01.wav"
    mdemo._write_pcm_wav(
        str(wav_16k),
        mdemo._synth_phrase_audio([440.0], 1.2, 16000),
        sample_rate=16000,
    )
    # Create an MP3 cue via ffmpeg
    tmp_wav = tmp_path / "src44.wav"
    mdemo._write_pcm_wav(
        str(tmp_wav),
        mdemo._synth_phrase_audio([523.25], 1.2, 44100),
        sample_rate=44100,
    )
    mp3_path = lines_dir / "line_02.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3_path)],
        check=True,
        capture_output=True,
    )

    script = {
        "episode_id": "mixed_audio_ep",
        "pages": [
            {
                "page_id": "p01",
                "lines": [
                    {"line_id": "01", "speaker": "Narrator", "en": "Line one", "zh": "第一句"},
                    {"line_id": "02", "speaker": "Fox", "en": "Line two", "zh": "第二句"},
                ],
            }
        ],
    }
    work_dir = ep_dir / "_build_tmp"
    norm_wav, _, cues, _, _, _ = asm.build_audio_and_timeline(str(ep_dir), str(work_dir), script)
    assert os.path.exists(norm_wav)
    assert len(cues) == 2
    assert 1.0 <= cues[0]["dur"] <= 1.4
    assert 1.0 <= cues[1]["dur"] <= 1.4


def test_audit_feed_delivery_fails_on_bad_video(tmp_path: Path) -> None:
    bad_mp4 = tmp_path / "bad_dark_silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=2:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(bad_mp4),
        ],
        check=True,
        capture_output=True,
    )
    res = _run_script("audit_feed_delivery.py", [str(bad_mp4), "--json"])
    assert res.returncode == 2
    report = json.loads(res.stdout)
    assert report["fail_count"] >= 2


def test_flow_route_example_dry_run_and_ip_fallback(tmp_path: Path) -> None:
    ep_dir = tmp_path / "route_ep"
    (ep_dir / "pages").mkdir(parents=True)
    script = {
        "episode_id": "route_ep",
        "pages": [
            {
                "page_id": "p01",
                "animated": True,
                "clip_duration_target": 6,
                "ip_sensitive_for_veo": False,
                "image_prompt": "Forest cabin",
                "motion_prompt": "Slow pan across fireflies.",
            },
            {
                "page_id": "p02",
                "animated": True,
                "clip_duration_target": 6,
                "ip_sensitive_for_veo": True,
                "refs": ["assets/char_a.png"],
                "image_prompt": "Hero close-up",
                "motion_prompt": "Gentle head nod.",
            },
        ],
    }
    (ep_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")

    records = fre.plan_episode_routes(str(ep_dir), project="proj-123", execute=False)
    # p01 image + p01 video + p02 image + p02 video
    assert len(records) == 4
    p02_img_cmd = records[2]["command"]
    assert "--prompt" not in p02_img_cmd
    assert p02_img_cmd[-1] == "Hero close-up"

    p01_vid = records[1]
    p02_vid = records[3]
    assert p01_vid["initial_model"] == "veo-lite"
    assert "--max-credits" in p01_vid["commands"][0]
    assert "closed-mouth" in p01_vid["commands"][0][-1].lower()
    assert p02_vid["initial_model"] == "omni-1.2-flash"

    # Mock PUBLIC_ERROR_IP_INPUT_IMAGE automatic fallback on p01
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="PUBLIC_ERROR_IP_INPUT_IMAGE")
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    fb_res = fre.run_i2v_with_fallback(
        script["pages"][0],
        str(ep_dir / "pages" / "p01.png"),
        str(ep_dir / "clips" / "p01_anim.mp4"),
        execute=True,
        runner=fake_runner,
    )
    assert fb_res["fallback_triggered"] is True
    assert fb_res["initial_model"] == "veo-lite"
    assert fb_res["final_model"] == "omni-1.2-flash"
    assert fb_res["returncode"] == 0
    assert len(calls) == 2


def test_gen_tts_lines_dry_run_and_skip_existing(tmp_path: Path) -> None:
    ep_dir = tmp_path / "tts_ep"
    lines_dir = ep_dir / "audio" / "lines"
    lines_dir.mkdir(parents=True)
    (lines_dir / "line_01.wav").write_bytes(b"placeholder")

    script = {
        "voices": {
            "Narrator": {"voice": "Vindemiatrix", "style": "Calm"},
            "Fox": {"voice": "Achird", "style": "Bright"},
        },
        "pages": [
            {
                "page_id": "p01",
                "lines": [
                    {"line_id": "01", "speaker": "Narrator", "en": "First line."},
                    {"line_id": "02", "speaker": "Fox", "en": "Second line."},
                ],
            }
        ],
    }
    jobs = gtts.collect_tts_jobs(str(ep_dir), script)
    res = gtts.run_tts_jobs(jobs, provider="gemini-cli", dry_run=True)
    assert res[0]["status"] == "skipped"
    assert res[1]["status"] == "dry_run"
    assert res[1]["command"][0] == "gemini-api"

    # Force single-line retake
    jobs_force = gtts.collect_tts_jobs(str(ep_dir), script, only_line="01", force=True)
    assert len(jobs_force) == 1
    res_force = gtts.run_tts_jobs(jobs_force, provider="edge-tts", dry_run=True)
    assert res_force[0]["status"] == "dry_run"
    assert res_force[0]["command"][0] == "edge-tts"

#!/usr/bin/env python3
"""Optional decoupled TTS line generator for storybook episodes.

Reads `voices` and `pages[].lines[]` from `script.json`, generates each line via
`gemini-api tts` or `edge-tts`, and normalizes output to 48kHz 16-bit mono WAV
under `audio/lines/line_<line_id>.wav`.

Notice: `assemble_storybook.py` is completely TTS-agnostic and accepts any WAV/MP3/M4A
in `audio/lines/`. You do NOT need this helper if you bring your own recordings or
another TTS provider.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional


def collect_tts_jobs(
    ep_dir: str,
    script: Dict[str, Any],
    only_line: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Collect per-line TTS jobs, skipping existing files unless force=True."""
    voices = script.get("voices") or {}
    lines_dir = os.path.join(ep_dir, "audio", "lines")
    jobs: List[Dict[str, Any]] = []

    for page in script.get("pages", []):
        for ln in page.get("lines") or []:
            lid = str(ln["line_id"])
            if only_line is not None and lid != str(only_line):
                continue
            speaker = ln.get("speaker", "Narrator")
            v_cfg = voices.get(speaker, {}) if isinstance(voices, dict) else {}
            voice_name = str(v_cfg.get("voice") or "Kore")
            style = str(ln.get("style_override") or v_cfg.get("style") or "").strip()
            text = str(ln.get("en") or ln.get("zh") or "").strip()
            out_wav = os.path.join(lines_dir, f"line_{lid}.wav")
            skip = os.path.exists(out_wav) and not force
            jobs.append({
                "line_id": lid,
                "speaker": speaker,
                "voice": voice_name,
                "style": style,
                "text": text,
                "out_wav": out_wav,
                "skip": skip,
            })
    return jobs


def build_tts_command(job: Dict[str, Any], provider: str, raw_out: str) -> List[str]:
    """Construct the external TTS CLI command for a single line."""
    text = job["text"]
    if provider == "gemini-cli":
        prompt = f"[{job['style']}] {text}" if job["style"] else text
        return ["gemini-api", "tts", prompt, "-v", job["voice"], "--out", raw_out]
    if provider == "edge-tts":
        return ["edge-tts", "--voice", job["voice"], "--text", text, "--write-media", raw_out]
    raise ValueError(f"Unsupported TTS provider: {provider}")


def run_tts_jobs(
    jobs: List[Dict[str, Any]],
    provider: str = "gemini-cli",
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run or preview TTS jobs and normalize produced audio to 48kHz mono WAV."""
    results: List[Dict[str, Any]] = []
    for job in jobs:
        if job["skip"]:
            results.append({"line_id": job["line_id"], "status": "skipped", "out_wav": job["out_wav"]})
            continue
        if not job["text"]:
            results.append({"line_id": job["line_id"], "status": "empty_text", "out_wav": job["out_wav"]})
            continue

        ext = ".mp3" if provider == "edge-tts" else ".wav"
        raw_preview = os.path.join(os.path.dirname(job["out_wav"]), f"_raw_{job['line_id']}{ext}")
        cmd = build_tts_command(job, provider, raw_preview)
        if dry_run:
            results.append({
                "line_id": job["line_id"],
                "status": "dry_run",
                "command": cmd,
                "out_wav": job["out_wav"],
            })
            continue

        os.makedirs(os.path.dirname(job["out_wav"]), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpd:
            raw_tmp = os.path.join(tmpd, f"raw_{job['line_id']}{ext}")
            real_cmd = build_tts_command(job, provider, raw_tmp)
            subprocess.run(real_cmd, check=True)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", raw_tmp,
                    "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
                    job["out_wav"],
                ],
                check=True,
                capture_output=True,
            )
        results.append({"line_id": job["line_id"], "status": "generated", "out_wav": job["out_wav"]})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional decoupled per-line TTS generator.")
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing script.json")
    parser.add_argument("--provider", choices=["gemini-cli", "edge-tts"], default="gemini-cli", help="TTS CLI backend")
    parser.add_argument("--line", default=None, help="Synthesize only a specific line_id")
    parser.add_argument("--force", action="store_true", help="Overwrite existing audio/lines/line_<id>.wav files")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without invoking TTS backend")
    args = parser.parse_args()

    ep_dir = os.path.abspath(args.episode_dir)
    with open(os.path.join(ep_dir, "script.json"), "r", encoding="utf-8") as f:
        script = json.load(f)

    jobs = collect_tts_jobs(ep_dir, script, only_line=args.line, force=args.force)
    results = run_tts_jobs(jobs, provider=args.provider, dry_run=args.dry_run)

    for r in results:
        if r["status"] == "dry_run":
            print(f"[DRY-RUN] line_{r['line_id']}: {shlex.join(r['command'])} -> {r['out_wav']}")
        else:
            print(f"[{r['status'].upper()}] line_{r['line_id']}: {r['out_wav']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

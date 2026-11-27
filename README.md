# flow-storybook-kit

[中文文档](README_CN.md)

**Assemble flicker-free `1080x1920 @ 30fps` vertical AI storybook videos with `0` GPU cost on your local machine.**

Born out of running real production episodes on Google Flow: after burning ~1,000 credits trying to generate continuous full-motion 27-second clips in one shot (where one bad lip or hand forced a full re-roll), we refactored to a **paginated hybrid architecture**:
- Static 9:16 storybook pages (`nano2` + 2K upscale = **0 video credits**)
- Only 2–3 emotional climax pages animated as silent clips (`veo-lite` / `omni-1.2-flash`, **10–23 credits** total per ~90s episode)
- **All post-production runs locally at 0 credits**: sub-pixel `cv2.warpAffine` Ken Burns, decoupled TTS + CC0 foley + ducked BGM, 2-pass `-15 LUFS` loudness, bilingual/single-language subtitle pills, and a 9-point mobile feed auditor.

---

## Why This Pipeline Exists

1. **Paginated beat-by-beat assembly instead of all-or-nothing clips**: If page 4 breaks, you only regenerate page 4.
2. **Fixes the `ffmpeg zoompan` 7.5Hz flicker**: `zoompan` steps at integer pixels on `1080p` stills, causing visible shimmer/pulsing on subtle slow zooms. `assemble_storybook.py` oversamples to `1620x2880` (`1.5x`) and uses cosine-eased floating-point `cv2.warpAffine`.
3. **Never pay credits twice for audio or subtitles**: Flow web editor exports max out at 720p, and in-editor Omni re-edits burn extra credits per tweak and risk drifting the illustration. Keep Flow strictly for silent visuals, and do 100% of audio/subtitles locally (`--reoverlay-only` re-burns subtitles in seconds).
4. **Hybrid `veo-lite` / `omni-1.2-flash` routing**: Route non-sensitive pages to `veo-lite` (5 credits/6s) and automatically fall back to `omni-1.2-flash` (3 credits/s) if Flow returns `PUBLIC_ERROR_IP_INPUT_IMAGE`, while appending a closed-mouth guard so clips never fight your TTS voiceover.

---

## Architecture

```text
  script.json (pages + lines + speaker styles)
       │
       ├──► Visuals (Optional: scripts/flow_route_example.py + gflow-cli)
       │       ├── Static pages: nano2 (0 video cr) -> 2K upscale (0 cr)
       │       └── 2-3 climax pages: veo-lite / omni-1.2-flash silent i2v
       │
       ├──► Voice & Audio (TTS-agnostic: bring ANY WAV/MP3/M4A, or scripts/gen_tts_lines.py)
       │       └── line_<id>.* (any sample rate) + optional foley + optional BGM
       │
       ▼
  scripts/assemble_storybook.py  (Local CPU, 0 credits)
       ├── 1.5x oversampled cv2.warpAffine Ken Burns + exact-final-frame clip hold
       ├── NumPy BGM speech-ducking + 2-pass EBU R128 loudnorm (-15 LUFS)
       └── Data-driven subtitle pills & presets (shorts / douyin / xiaohongshu / clean)
       │
       ▼
  scripts/audit_feed_delivery.py  (9-point mobile feed readiness gate)
       └── 1080x1920, 30fps, 6-14 Mbps, -16.5..-13 LUFS, LRA <= 8.5, YAVG >= 60
```

---

## Quick Start (Zero API Keys, Zero Credits)

Requires **Python 3.9+** and **FFmpeg** (`brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg` on Ubuntu).

```bash
git clone https://github.com/danugot/flow-storybook-kit.git
cd flow-storybook-kit

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Generate a synthetic 3-page episode (includes 24fps anim clip + mixed 24k/44.1k/48k audio)
python scripts/make_demo.py --out-dir demo_episode

# 2. Assemble 1080x1920 @ 30fps MP4s locally
python scripts/assemble_storybook.py \
  --episode-dir demo_episode \
  --prefix demo \
  --presets shorts,xiaohongshu,douyin

# 3. Run the 9-point mobile feed readiness check
python scripts/audit_feed_delivery.py demo_episode/demo_shorts_ready.mp4
```

You can also inspect the dry-run routing and TTS helpers without spending any credits:

```bash
python scripts/flow_route_example.py --episode-dir demo_episode
python scripts/gen_tts_lines.py --episode-dir demo_episode --dry-run
pytest -v
```

---

## Using Your Own Assets (`gflow-cli` + Any TTS)

1. Copy [`examples/sample_script.json`](examples/sample_script.json) into `episodes/my_story/script.json`.
2. **TTS is completely decoupled**: put your voice files (from Gemini TTS, ElevenLabs, Edge-TTS, or human recording, in `.wav`/`.mp3`/`.m4a` at any sample rate) in `episodes/my_story/audio/lines/line_<line_id>.<ext>`, or run the optional helper:
   ```bash
   python scripts/gen_tts_lines.py --episode-dir episodes/my_story --provider gemini-cli
   ```
3. Preview or run Google Flow generation serially via [`gflow-cli`](https://github.com/ffroliva/gflow-cli):
   ```bash
   # Preview exact commands (dry-run by default)
   python scripts/flow_route_example.py --episode-dir episodes/my_story
   # Execute serially (never run two gflow processes concurrently on the same browser profile)
   python scripts/flow_route_example.py --episode-dir episodes/my_story --upscale-2k --execute
   ```
4. Assemble & audit:
   ```bash
   python scripts/assemble_storybook.py --episode-dir episodes/my_story --prefix ep01 --presets clean,shorts
   python scripts/audit_feed_delivery.py episodes/my_story/ep01_shorts_ready.mp4
   ```
   Tweak subtitle copy? Edit `script.json` and run `--reoverlay-only` to re-burn overlays in seconds without re-rendering Ken Burns segments.

---

## Honest Boundaries & Disclaimers

- **`gflow-cli` disclaimer**: `flow_route_example.py` targets [`ffroliva/gflow-cli`](https://github.com/ffroliva/gflow-cli) (MIT), a Playwright-based browser automation tool—not an official Google Flow public SLA API. Check Google Flow's Terms of Service and credit pricing before running `--execute`.
- **TTS is optional**: `assemble_storybook.py` does not lock you into any voice provider and makes zero external API calls.
- **Keep a human Eye & Ear Gate**: `audit_feed_delivery.py` verifies objective technical thresholds (resolution, FPS, bitrate, LUFS, luma). Always spot-check final frames and listen to start/middle/end audio before publishing to catch character drift or TTS breathing artifacts.

## License

MIT

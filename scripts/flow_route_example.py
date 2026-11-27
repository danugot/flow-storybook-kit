#!/usr/bin/env python3
"""Hybrid Google Flow routing template using gflow-cli.

Demonstrates the paginated cost-saving pattern:
1. Generate 9:16 pages with nano2 (0 video credits) and optionally upscale to 2K.
2. Animate ONLY selected emotional/climax pages as silent I2V clips:
   - Default to `veo-lite` (low credit cost)
   - Route `ip_sensitive_for_veo: true` pages directly to `omni-1.2-flash`
   - Automatically fall back to `omni-1.2-flash` if `veo-lite` returns
     `PUBLIC_ERROR_IP_INPUT_IMAGE`
   - Append a closed-mouth guard so generated video never competes with external TTS
3. Execute strictly serially (gflow uses a single Playwright browser profile).

Dry-run is enabled by default unless `--execute` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

SILENT_GUARD_SUFFIX = " Silent characters, closed-mouth calm expression, no speech, ambient sound only."


def build_image_command(
    page: Dict[str, Any],
    out_file: str,
    project: Optional[str] = None,
    model: str = "nano2",
    aspect: str = "9:16",
) -> List[str]:
    """Build gflow image i2i/t2i command with positional PROMPT."""
    prompt = (page.get("image_prompt") or f"Vertical 9:16 storybook scene for page {page['page_id']}.").strip()
    refs = page.get("refs") or []
    entities = page.get("reference_entities") or []

    if refs:
        cmd = ["gflow", "image", "i2i", "--model", model, "--aspect", aspect]
        for ref in refs:
            cmd.extend(["--ref", str(ref)])
        for ent in entities:
            cmd.extend(["--reference-entity", str(ent)])
    else:
        cmd = ["gflow", "image", "t2i", "--model", model, "--aspect", aspect]

    if project:
        cmd.extend(["--project", project])
    cmd.extend(["--out-file", out_file, "--json", prompt])
    return cmd


def build_upscale_command(
    media_id: str,
    out_dir: str,
    scale: str = "2k",
    project: Optional[str] = None,
) -> List[str]:
    """Build gflow image upscale command (0 video credits)."""
    cmd = ["gflow", "image", "upscale", media_id, "--scale", scale, "--out", out_dir]
    if project:
        cmd.extend(["--project", project])
    return cmd


def select_initial_video_model(page: Dict[str, Any]) -> Tuple[str, int]:
    """Select video model and pre-submit credit ceiling for an animated page."""
    if page.get("ip_sensitive_for_veo"):
        return "omni-1.2-flash", 25
    return "veo-lite", 10


def build_motion_prompt(page: Dict[str, Any]) -> str:
    """Append closed-mouth silent guard unless disabled on the page."""
    base = (page.get("motion_prompt") or f"Subtle storybook motion for page {page['page_id']}.").strip()
    if page.get("silent_guard", True) and "closed-mouth" not in base.lower():
        return base + SILENT_GUARD_SUFFIX
    return base


def build_i2v_command(
    page: Dict[str, Any],
    initial_frame: str,
    out_file: str,
    model: str,
    max_credits: int,
    project: Optional[str] = None,
    aspect: str = "9:16",
) -> List[str]:
    """Build gflow video i2v command."""
    dur = int(page.get("clip_duration_target", 6))
    prompt = build_motion_prompt(page)
    cmd = [
        "gflow", "video", "i2v",
        "--initial-frame", initial_frame,
        "--model", model,
        "--aspect", aspect,
        "--duration", str(dur),
        "--max-credits", str(max_credits),
        "--out-file", out_file,
    ]
    if project:
        cmd.extend(["--project", project])
    cmd.append(prompt)
    return cmd


def find_existing_page_image(pages_dir: str, page_id: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = os.path.join(pages_dir, page_id + ext)
        if os.path.exists(cand):
            return cand
    return None


def run_i2v_with_fallback(
    page: Dict[str, Any],
    initial_frame: str,
    out_file: str,
    project: Optional[str] = None,
    execute: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Dict[str, Any]:
    """Plan or run serial I2V with automatic veo-lite -> omni-1.2-flash fallback."""
    model, max_credits = select_initial_video_model(page)
    cmd = build_i2v_command(page, initial_frame, out_file, model, max_credits, project=project)
    result: Dict[str, Any] = {
        "page_id": page["page_id"],
        "initial_model": model,
        "final_model": model,
        "fallback_triggered": False,
        "commands": [cmd],
        "executed": execute,
        "returncode": 0,
    }
    if not execute:
        return result

    env = dict(os.environ)
    env["GFLOW_CLI_UPDATE_CHECK"] = "0"
    proc = runner(cmd, env=env, capture_output=True, text=True)
    result["returncode"] = proc.returncode
    combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0 and model == "veo-lite" and "PUBLIC_ERROR_IP_INPUT_IMAGE" in combined_out:
        fb_cmd = build_i2v_command(page, initial_frame, out_file, "omni-1.2-flash", 25, project=project)
        result["commands"].append(fb_cmd)
        result["fallback_triggered"] = True
        result["final_model"] = "omni-1.2-flash"
        fb_proc = runner(fb_cmd, env=env, capture_output=True, text=True)
        result["returncode"] = fb_proc.returncode

    return result


def plan_episode_routes(
    ep_dir: str,
    project: Optional[str] = None,
    upscale_2k: bool = False,
    execute: bool = False,
) -> List[Dict[str, Any]]:
    """Build or run serial image + animated-page routing plan for an episode."""
    script_path = os.path.join(ep_dir, "script.json")
    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    pages_dir = os.path.join(ep_dir, "pages")
    clips_dir = os.path.join(ep_dir, "clips")
    if execute:
        os.makedirs(pages_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)

    env = dict(os.environ)
    env["GFLOW_CLI_UPDATE_CHECK"] = "0"
    records: List[Dict[str, Any]] = []

    for page in script.get("pages", []):
        pid = page["page_id"]
        existing_img = find_existing_page_image(pages_dir, pid)
        img_target = existing_img or os.path.join(pages_dir, f"{pid}.png")

        if page.get("image_prompt") and not existing_img:
            img_cmd = build_image_command(page, img_target, project=project)
            rec: Dict[str, Any] = {"page_id": pid, "step": "image", "command": img_cmd, "executed": execute}
            if execute:
                proc = subprocess.run(img_cmd, env=env, capture_output=True, text=True)
                rec["returncode"] = proc.returncode
                if upscale_2k and proc.returncode == 0:
                    try:
                        payload = json.loads(proc.stdout)
                        mid = payload.get("media_id") or payload.get("id")
                        if mid:
                            up_cmd = build_upscale_command(str(mid), pages_dir, scale="2k", project=project)
                            rec["upscale_command"] = up_cmd
                            subprocess.run(up_cmd, env=env, check=True, capture_output=True, text=True)
                    except (json.JSONDecodeError, subprocess.CalledProcessError):
                        pass
            records.append(rec)

        if page.get("animated"):
            clip_out = os.path.join(clips_dir, f"{pid}_anim.mp4")
            if execute and os.path.exists(clip_out):
                records.append({"page_id": pid, "step": "video_i2v", "skipped": True, "reason": "clip_exists"})
                continue
            i2v_res = run_i2v_with_fallback(page, img_target, clip_out, project=project, execute=execute)
            i2v_res["step"] = "video_i2v"
            records.append(i2v_res)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid gflow-cli routing template (dry-run by default).")
    parser.add_argument("--episode-dir", required=True, help="Episode directory containing script.json")
    parser.add_argument("--project", default=None, help="Optional Flow project UUID")
    parser.add_argument("--upscale-2k", action="store_true", help="Include 2K upscale step after image generation")
    parser.add_argument("--execute", action="store_true", help="Actually invoke gflow serially (default is dry-run)")
    parser.add_argument("--json", action="store_true", help="Print plan/result as JSON")
    args = parser.parse_args()

    ep_dir = os.path.abspath(args.episode_dir)
    records = plan_episode_routes(
        ep_dir,
        project=args.project,
        upscale_2k=args.upscale_2k,
        execute=args.execute,
    )

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        mode_lbl = "EXECUTE" if args.execute else "DRY-RUN (no credits spent)"
        print(f"=== Flow Hybrid Route Plan [{mode_lbl}] ===")
        for rec in records:
            pid = rec["page_id"]
            if rec.get("skipped"):
                print(f"[{pid}] SKIP {rec['step']}: {rec.get('reason')}")
                continue
            if "command" in rec:
                print(f"[{pid}] {rec['step']}: {shlex.join(rec['command'])}")
            for cmd in rec.get("commands", []):
                tag = "fallback" if rec.get("fallback_triggered") and cmd is rec["commands"][-1] else rec["step"]
                print(f"[{pid}] {tag} ({rec.get('final_model')}): {shlex.join(cmd)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

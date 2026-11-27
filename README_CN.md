# flow-storybook-kit

[English README](README.md)

**零本地 GPU 成本，在本地把 AI 绘本分页插画 + 少量点睛动态片段，拼装成无闪烁 `1080×1920 @ 30fps` 竖屏成片。**

这套工具源于真实跑 Google Flow 的踩坑重构：最早跑全动态连续短片时，一镜出瑕疵就得整段重拍，废片率很高。改成 **「分页混动架构」** 后：
- 大多数页面用静态 9:16 绘本页（`nano2` 生图 + 原生 2K 放大 = **0 视频积分**）
- 全片只挑 2–3 个情绪拐点做无声活页动态（`veo-lite` / `omni-1.2-flash` 混用路由，90 秒短片仅 **10–23 积分**）
- **配音、拟音、BGM 闪避、响度归一、字幕压制全部收归本地 0 积分跑完**，并附带 9 项移动端信息流交付体检脚本。

---

## 核心解决什么问题

1. **防 `ffmpeg zoompan` 7.5Hz 闪烁**：直接在 1080p 静态图上用 `zoompan` 会因为整数像素步进产生明显呼吸闪烁。本库先超采样至 `1620×2880`（`1.5x`），再用 `cv2.warpAffine` 做余弦缓动浮点运镜，消除画面抖动。
2. **绝不在在线编辑器里花冤枉分改声音/字幕**：Flow 网页时间线导出上限 720p，且用 Omni 局部重绘修改音乐/字幕每次都要扣分还容易漂移原画。Flow 只输出无声底片，声音和字幕全部本地可控，`--reoverlay-only` 改字幕秒出新片。
3. **`veo-lite` / `omni-1.2-flash` 混用路由与自动兜底**：普通页走低积分 `veo-lite`，遇到 `PUBLIC_ERROR_IP_INPUT_IMAGE` 自动降级切 `omni-1.2-flash`；提示词强制追加闭嘴尾句，避免画面人物自己张嘴和外挂配音打架。
4. **完全解耦的音频摄取**：无论你用 Gemini TTS、Edge-TTS、ElevenLabs 还是自己录音，任何采样率（16k/24k/44.1k/48k）或格式（`.wav/.mp3/.m4a/.flac/.ogg`）都会经 `ffmpeg` 自动统一为 48kHz 立体声混音并做双遍 `-15 LUFS` 响度标准化。

---

## 零成本快速体验（无需 API Key、不花积分）

前置依赖：**Python 3.9+** 与 **FFmpeg**（macOS 执行 `brew install ffmpeg`，Ubuntu 执行 `sudo apt-get install ffmpeg`）。

```bash
git clone https://github.com/<your-username>/flow-storybook-kit.git
cd flow-storybook-kit

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. 自动生成 3 页测试绘本（含 1 段 24fps 合成活页与 24k/44.1k/48k 混合采样率音频）
python scripts/make_demo.py --out-dir demo_episode

# 2. 本地拼装 1080x1920 @ 30fps 成片（支持 shorts / douyin / xiaohongshu / clean）
python scripts/assemble_storybook.py \
  --episode-dir demo_episode \
  --prefix demo \
  --presets shorts,xiaohongshu,douyin

# 3. 跑一遍 9 项移动端交付体检
python scripts/audit_feed_delivery.py demo_episode/demo_shorts_ready.mp4
```

也可以零消耗预览 `gflow` 路由和可选 TTS 命令清单，以及运行完整单测：

```bash
python scripts/flow_route_example.py --episode-dir demo_episode
python scripts/gen_tts_lines.py --episode-dir demo_episode --dry-run
pytest -v
```

---

## 目录与脚本说明

- `scripts/assemble_storybook.py`：核心本地拼装引擎（`1.5x` 浮点 Ken Burns、活页末帧无缝定格、NumPy 语音包络闪避 BGM、2-pass `-15 LUFS` 响度、支持双语或纯英文字幕胶囊、可选结尾复习卡、`--reoverlay-only` 秒重压字幕）。
- `scripts/audit_feed_delivery.py`：移动端竖屏交付 9 项体检（`1080×1920`、`30fps`、码率、响度 `LUFS`、`LRA`、真峰值、平均亮度 `YAVG`、首帧亮度、黑帧检测），`0 FAIL` 返回 `0`，存在 `FAIL` 返回 `2`。
- `scripts/flow_route_example.py`：配合开源 [`gflow-cli`](https://github.com/ffroliva/gflow-cli) 的分页混用路由模板（默认 `--dry-run`，需显式传 `--execute` 才串行调起浏览器执行）。
- `scripts/gen_tts_lines.py`：完全可选的逐行配音助手（支持 `gemini-cli` 与 `edge-tts`），不使用它也完全不影响主拼装流水线。
- `scripts/make_demo.py`：零依赖测试数据生成器。

---

## 坦诚边界提示

- `gflow-cli` 是第三方开源（MIT）Playwright 浏览器自动化工具，并非 Google 官方稳定接口；实际运行 `--execute` 前请先在 `--dry-run` 下核对积分预算与官方服务条款。
- 自动化体检脚本能拦截分辨率、响度和亮度等硬伤，但**无法替代人工的 Eye/Ear Gate**（例如人物跨页特征漂移、TTS 句首喘息声、拟音与画面贴合度仍建议抽帧与盲听复核）。

## 开源协议

MIT

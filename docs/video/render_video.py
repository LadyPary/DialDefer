"""Render docs/video/teaser.html to an MP4, frame by frame.

The page exposes window.DURATION (seconds) and window.renderAt(t), which draws
the frame at time t. Each frame is screenshotted from headless Chrome and piped
to ffmpeg, so the output is smooth regardless of machine speed.

Setup (outside the project venv is fine):
    python3 -m venv /tmp/vvenv && /tmp/vvenv/bin/pip install playwright imageio-ffmpeg
Usage (from repo root):
    /tmp/vvenv/bin/python docs/video/render_video.py                       # 1920x1080 -> docs/static/videos/dialdefer.mp4
    /tmp/vvenv/bin/python docs/video/render_video.py --frames 2.5,15.5,26.5 --out /tmp/frames   # stills to check
The layout is designed for 16:9; other sizes are letterboxed.
Uses the installed Google Chrome (channel="chrome"); no browser download needed.
"""
import argparse
import os
import pathlib
import subprocess

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(HERE / "teaser.html"))
    ap.add_argument("--out", default=str(HERE.parent / "static/videos/dialdefer.mp4"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--poster", default=None, help="also save a PNG of the frame at this time (s)")
    ap.add_argument("--poster-out", default=str(HERE.parent / "static/videos/dialdefer_poster.jpg"))
    ap.add_argument("--frames", default=None, help="only save PNGs at these comma-separated times (s) to --out dir")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    url = pathlib.Path(args.html).resolve().as_uri() + f"?w={w}&h={h}"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page.goto(url)
        page.wait_for_function("window.renderAt && document.fonts.status === 'loaded'")
        page.evaluate("document.fonts.ready")
        dur = page.evaluate("window.DURATION")

        if args.frames:
            os.makedirs(args.out, exist_ok=True)
            for t in [float(x) for x in args.frames.split(",")]:
                page.evaluate(f"window.renderAt({t})")
                page.screenshot(path=os.path.join(args.out, f"frame_{t:05.2f}.png"))
            browser.close()
            return

        if args.poster is not None:
            page.evaluate(f"window.renderAt({float(args.poster)})")
            page.screenshot(path=args.poster_out, type="jpeg", quality=88)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        ff = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "image2pipe", "-framerate", str(args.fps), "-c:v", "png", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20",
            "-movflags", "+faststart", args.out,
        ], stdin=subprocess.PIPE)
        n = int(round(dur * args.fps))
        for i in range(n):
            page.evaluate(f"window.renderAt({i / args.fps})")
            ff.stdin.write(page.screenshot(type="png"))
            if i % args.fps == 0:
                print(f"\r{i / args.fps:5.1f}s / {dur:.1f}s", end="", flush=True)
        ff.stdin.close()
        ff.wait()
        browser.close()
    print(f"\nwrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()

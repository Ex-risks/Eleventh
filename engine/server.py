"""ELEVENTH engine server.

Serves:
  - The web page (HTTP GET /)
  - Static assets (HTTP GET /*)
  - WebSocket at ws://localhost:7979

WebSocket protocol (JSON messages):
  Client → Server:
    {"type": "file", "name": "<filename>", "data": "<base64-encoded bytes>"}
    {"type": "pause"}
    {"type": "resume"}
    {"type": "save"}   → server replies with {"type": "saved", "path": "..."}

  Server → Client:
    {"type": "training", "step": N, "total": N, "loss": F, "sample": "<text>"}
    {"type": "frame",    "cells": [{"char": "x", "masked": false}, ...]}
    {"type": "done"}     — page fully revealed; client holds, then sends nothing
    {"type": "saved",    "path": "..."}
    {"type": "error",    "message": "..."}
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import websockets

# Add engine dir to path regardless of where we're invoked from.
ENGINE_DIR = Path(__file__).parent
WEB_DIR = ENGINE_DIR.parent / "web"
sys.path.insert(0, str(ENGINE_DIR))

import config as C

WS_HOST = "localhost"
WS_PORT = 7979
HTTP_PORT = 7878

# Sentinel so the generation loop can check for pause/stop.
_state = {"paused": False, "stop_gen": False, "model": None, "last_frame": None}


# ── HTTP server ────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence request logs

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            filepath = WEB_DIR / "index.html"
        else:
            filepath = WEB_DIR / path.lstrip("/")

        if not filepath.exists() or not filepath.is_file():
            self.send_response(404); self.end_headers(); return

        mime = "text/html"
        if filepath.suffix == ".css":  mime = "text/css"
        elif filepath.suffix == ".js": mime = "application/javascript"
        elif filepath.suffix in (".woff", ".woff2"): mime = "font/woff2"

        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _run_http():
    srv = HTTPServer((WS_HOST, HTTP_PORT), Handler)
    srv.serve_forever()


# ── WebSocket handler ──────────────────────────────────────────────────────────

async def handle(ws):
    print(f"[ws] client connected")
    # One client at a time is fine for a single-user installation.
    try:
        await _session(ws)
    except websockets.exceptions.ConnectionClosed:
        print("[ws] client disconnected")
    except Exception as e:
        print(f"[ws] error: {e}")
        import traceback; traceback.print_exc()
        try:
            await ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def _session(ws):
    from train import train
    from generate import generate_page

    model = _state.get("model")
    _state["stop_gen"] = False

    async def send(obj):
        await ws.send(json.dumps(obj))

    # Listen for incoming messages while generation may be running.
    async def reader():
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t == "file":
                # New file dropped: stop any running generation, retrain.
                _state["stop_gen"] = True
                _state["model"] = None
                _state["paused"] = False
                name = msg.get("name", "dropped.txt")
                data = base64.b64decode(msg["data"])
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="eleventh_")
                tmp.write(data); tmp.close()
                asyncio.get_running_loop().create_task(_train_and_dream(ws, tmp.name, name))
            elif t == "pause":
                _state["paused"] = True
            elif t == "resume":
                _state["paused"] = False
            elif t == "save":
                frame = _state.get("last_frame")
                if frame:
                    text = "".join(c["char"] for c in frame)
                    save_dir = Path.home() / "Desktop" / "ELEVENTH pages"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fname = save_dir / f"page_{int(time.time())}.txt"
                    fname.write_text(text, encoding="utf-8")
                    await send({"type": "saved", "path": str(fname)})

    # Start reader; it drives everything via tasks.
    await reader()


async def _train_and_dream(ws, corpus_path: str, display_name: str):
    from train import train
    from generate import generate_page

    _state["stop_gen"] = False

    async def send(obj):
        try:
            await ws.send(json.dumps(obj))
        except Exception:
            pass

    # Training happens in a thread (MLX blocks the GIL).
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def cb(step, total, loss, sample):
        loop.call_soon_threadsafe(q.put_nowait, ("training", step, total, loss, sample))

    def do_train():
        try:
            m, tok = train(corpus_path, status_callback=cb)
            loop.call_soon_threadsafe(q.put_nowait, ("done", (m, tok)))
        except Exception as e:
            import traceback; traceback.print_exc()
            loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))

    thread = threading.Thread(target=do_train, daemon=True)
    thread.start()

    model = tokenizer = None
    while model is None:
        item = await q.get()
        if item[0] == "training":
            _, step, total, loss, sample = item
            await send({"type": "training", "step": step, "total": total,
                        "loss": round(loss, 4), "sample": sample})
        elif item[0] == "done":
            model, tokenizer = item[1]
        elif item[0] == "err":
            await send({"type": "error", "message": item[1]})
            return

    _state["model"] = model
    os.unlink(corpus_path)

    # Flush any remaining training messages.
    while not q.empty():
        await q.get_nowait()

    # Endless dreaming loop.
    while True:
        if _state["stop_gen"]:
            return
        if _state["paused"]:
            await asyncio.sleep(0.2)
            continue

        gen = generate_page(model, tokenizer)
        for frame in gen:
            if _state["stop_gen"] or _state.get("model") is not model:
                return
            while _state["paused"]:
                await asyncio.sleep(0.2)
            _state["last_frame"] = frame
            await send({"type": "frame", "cells": frame})
            # Gentle pace: let the frontend control visual timing; server just
            # streams at a steady low rate (the frontend queues and paces display).
            await asyncio.sleep(0.05)

        await send({"type": "done"})
        # Brief pause after each completed page before starting the next.
        await asyncio.sleep(3.5)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # Start HTTP server in a background thread.
    t = threading.Thread(target=_run_http, daemon=True)
    t.start()
    print(f"[http] serving web page at http://{WS_HOST}:{HTTP_PORT}")
    print(f"[ws]   WebSocket at ws://{WS_HOST}:{WS_PORT}")

    async with websockets.serve(handle, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())

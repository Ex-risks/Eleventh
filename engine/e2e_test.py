"""End-to-end test: start server subprocess, upload a file over WS, watch training + frames."""
import asyncio, base64, json, os, sys, tempfile, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))

async def run():
    import websockets
    ROOT = os.path.dirname(os.path.dirname(__file__))

    # Start a fresh server with short training steps.
    env = os.environ.copy()
    env["ELEVENTH_MIN_STEPS"] = "200"
    env["ELEVENTH_MAX_STEPS"] = "200"
    proc = subprocess.Popen(
        ["uv", "run", "python", "engine/server.py"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Wait for server to be ready.
    import urllib.request, urllib.error
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:7878", timeout=1)
            break
        except Exception:
            await asyncio.sleep(0.5)

    # Build a tiny corpus file.
    corpus = ("The wolf moves through the forest, silent. A step. A pause. "
              "Wind carries the scent of pine and something older. "
              "The pack waits at the ridge.\n\n") * 200
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(corpus); fname = f.name
    b64 = base64.b64encode(open(fname,"rb").read()).decode()

    training_msgs = []
    frame_msgs    = []
    done_received = False

    async with websockets.connect("ws://localhost:7979") as ws:
        await ws.send(json.dumps({"type": "file", "name": "test.txt", "data": b64}))

        deadline = time.time() + 600  # 10 min max
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                print("TIMEOUT waiting for message"); break
            msg = json.loads(raw)
            t = msg["type"]
            if t == "training":
                training_msgs.append(msg)
                print(f"  training step={msg['step']}/{msg['total']} loss={msg['loss']:.3f}")
            elif t == "frame":
                frame_msgs.append(msg)
                if len(frame_msgs) % 10 == 1:
                    n_masked = sum(1 for c in msg["cells"] if c["masked"])
                    print(f"  frame #{len(frame_msgs)}  masked={n_masked}/{len(msg['cells'])}")
                if len(frame_msgs) >= 30:
                    break
            elif t == "done":
                done_received = True
                print("  page complete (done received)")
                break
            elif t == "error":
                print(f"  ERROR: {msg['message']}"); break

    proc.terminate()
    os.unlink(fname)

    print(f"\ntraining callbacks: {len(training_msgs)}")
    print(f"frame messages:     {len(frame_msgs)}")
    print(f"done received:      {done_received}")

    assert len(training_msgs) >= 1, "no training callbacks"
    assert len(frame_msgs) >= 10,  "too few frames"

    # Verify masking decreases over frames.
    first_masked = sum(1 for c in frame_msgs[0]["cells"] if c["masked"])
    last_masked  = sum(1 for c in frame_msgs[-1]["cells"] if c["masked"])
    print(f"masks: first frame={first_masked}, last frame={last_masked}")
    assert last_masked < first_masked, "denoising did not reduce masks"

    # Verify cells have the right structure.
    sample_cell = frame_msgs[0]["cells"][0]
    assert "char" in sample_cell and "masked" in sample_cell

    print("\nE2E TEST PASSED")

asyncio.run(run())

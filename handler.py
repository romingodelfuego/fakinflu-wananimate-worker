#!/usr/bin/env python3
"""fakinflu — worker wananimate : Wan 2.2 Animate (14B via ComfyUI). Route remplacement.

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression et apercu via /stream (return_aggregate_stream = False).
Meme squelette que scail/serverless/handler.py (chaque worker est autonome).

Entree (contrat transfer.py, contracts/wananimate.input.schema.json) :
  {"run_id", "r2": bool,
   "inputs":  signal control (defaut) :
                {"control": video de controle du prep (control_video.mp4 : pose/depth/facemesh),
                 "mask":    option, masque video du personnage (mask.mp4, hors mode fast)}
              signal pose (v1.2.0) :
                {"source":  video d'origine (prep/input.mp4) -> squelette DWPose + decor d'origine,
                 "mask":    masque video du personnage (prep/mask.mp4), obligatoire}
              + toujours {"ref": photo du personnage (PNG/JPEG)},
   "outputs": {"video": {put_url, key, max_bytes} | {max_bytes}},
   "outputs_json": {put_url, key} | null,
   "params":  {"preset": "draft|final", "resolution": int, "prompt": str, "negative": str,
               "seed": int|null, "precision": "int8|bf16"?, "signal": "control|pose"?,
               "mask_expand": int?, "fps": number?, "max_length": int?, "probe": bool?}}
"""
import glob, json, os, queue, re, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event
from workflow import build_workflow, plan, ram_bytes, required_inputs, signal_of, OUTPUT_FILES, OUTPUT_NAMES

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("WANANIMATE_JOB_TIMEOUT_S", "3500"))
IMAGES_RAM_SHARE = float(os.environ.get("WANANIMATE_IMAGES_RAM_SHARE", "0.30"))
IMAGES_RAM_FALLBACK_GB = float(os.environ.get("WANANIMATE_IMAGES_RAM_GB", "12"))
BF16_MIN_VRAM_GB = float(os.environ.get("WANANIMATE_BF16_MIN_VRAM_GB", "40"))


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI unreachable at {COMFY} after {max_s}s")


def _ws_progress(client_id, q, stop):
    try:
        import websocket
    except ImportError:
        return
    url = COMFY.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    try:
        ws = websocket.create_connection(url, timeout=5)
    except Exception:
        return
    ws.settimeout(1)
    while not stop.is_set():
        try:
            msg = ws.recv()
        except Exception:
            continue
        if not isinstance(msg, str):
            continue
        try:
            m = json.loads(msg)
        except ValueError:
            continue
        if m.get("type") in ("progress", "executing"):
            q.put(m)
    ws.close()


def run_workflow(wf):
    """Generateur : soumet le graphe, emet la progression, renvoie (secondes) via StopIteration."""
    client_id = uuid.uuid4().hex
    q, stop = queue.Queue(), threading.Event()
    th = threading.Thread(target=_ws_progress, args=(client_id, q, stop), daemon=True)
    th.start()
    try:
        try:
            r = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the graph (HTTP {e.code}): {e.read().decode()[:1500]}")
        pid = r["prompt_id"]
        t0, last_poll = time.time(), 0.0
        names = {k: v.get("_meta", {}).get("title") or v["class_type"] for k, v in wf.items()}
        while True:
            if time.time() - t0 > JOB_TIMEOUT:
                _http("POST", "/interrupt", {})
                raise RuntimeError(f"ComfyUI timeout ({JOB_TIMEOUT}s)")
            try:
                m = q.get(timeout=1)
                d = m.get("data", {})
                if m["type"] == "executing" and d.get("node"):
                    yield {"type": "progress", "step": names.get(d["node"], d["node"])}
                elif m["type"] == "progress" and d.get("max"):
                    yield {"type": "progress", "step": names.get(d.get("node"), "compute"),
                           "pct": round(100 * d["value"] / d["max"]), "frame": d["value"]}
            except queue.Empty:
                pass
            if time.time() - last_poll < 2:
                continue
            last_poll = time.time()
            try:
                h = _http("GET", f"/history/{pid}")
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as e:
                raise RuntimeError(f"ComfyUI stopped mid-compute ({e.reason}): {oom_report()}") from e
            if pid not in h:
                continue
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1] if msgs else st
                if isinstance(detail, dict):
                    detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
                raise RuntimeError(f"ComfyUI execution error: {json.dumps(detail, ensure_ascii=False)[:800]}")
            if st.get("completed"):
                return time.time() - t0
    finally:
        stop.set()


def collect(prefix):
    base = os.path.join(COMFY_DIR, "output", prefix)
    found = {}
    for role, pattern in OUTPUT_FILES.items():
        hits = sorted(glob.glob(os.path.join(base, pattern)))
        if hits:
            found[role] = hits[-1]
    return found


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", path], capture_output=True, text=True, timeout=120)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den or 1), "frames": int(s.get("nb_read_packets") or 0)}


def thumb(video, t=1.0):
    dst = video + ".jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "5", dst], timeout=60, check=True)
    with open(dst, "rb") as f:
        return f.read()


def memory_limit_bytes():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                v = f.read().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 1 << 60:
            return int(v)
    return None


def _cgroup(name):
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            with open(os.path.join(base, name)) as f:
                return f.read()
        except OSError:
            continue
    return None


def oom_report():
    events = _cgroup("memory.events") or ""
    kills = next((int(l.split()[1]) for l in events.splitlines() if l.startswith("oom_kill ")), None)
    peak = _cgroup("memory.peak") or _cgroup("memory.max_usage_in_bytes")
    limit = memory_limit_bytes()
    gb = lambda b: f"{int(b) / 1024 ** 3:.1f} GB"
    mem = f"peak {gb(peak.strip())}" if peak and peak.strip().isdigit() else "peak unknown"
    mem += f", limit {gb(limit)}" if limit else ", limit unknown"
    if kills:
        return (f"killed by the kernel out of memory (oom_kill={kills}, {mem}). Check COMFY_EXTRA_ARGS "
                f"(current: {os.environ.get('COMFY_EXTRA_ARGS', '')!r}), otherwise lower resolution")
    return f"no oom_kill in the cgroup ({mem}): likely native crash (worker RunPod logs)"


def _gb(b):
    return round(int(b) / 1024 ** 3, 2) if b not in (None, "") and str(b).strip().isdigit() else None


def probe_report():
    stats = {}
    try:
        stats = _http("GET", "/system_stats", timeout=10)
    except Exception as e:
        stats = {"error": f"{type(e).__name__}: {e}"}
    disk = {}
    for path in ("/", COMFY_DIR):
        try:
            st = os.statvfs(path)
            disk[path] = {"free_gb": round(st.f_bavail * st.f_frsize / 1024 ** 3, 2), "total_gb": round(st.f_blocks * st.f_frsize / 1024 ** 3, 2)}
        except OSError:
            pass
    system = stats.get("system") or {}
    return {
        "cgroup": {"limit_gb": _gb(memory_limit_bytes()), "peak_gb": _gb((_cgroup("memory.peak") or "").strip())},
        "comfyui": {"version": system.get("comfyui_version"), "pytorch": system.get("pytorch_version"),
                    "argv": system.get("argv"), "ram_total_gb": _gb(system.get("ram_total")),
                    "devices": [{k: d.get(k) for k in ("name", "type", "vram_total", "vram_free")} for d in stats.get("devices") or []],
                    "error": stats.get("error")},
        "disk": disk,
        "env": {"COMFY_EXTRA_ARGS": os.environ.get("COMFY_EXTRA_ARGS", "")},
    }


def check_vram(precision):
    try:
        devices = _http("GET", "/system_stats", timeout=10).get("devices") or []
    except Exception:
        return None
    vram = max((d.get("vram_total") or 0 for d in devices), default=0) / 1024 ** 3
    if precision == "bf16" and 0 < vram < BF16_MIN_VRAM_GB:
        raise ValueError(f"bf16 precision on {vram:.0f} GB of VRAM: needs {BF16_MIN_VRAM_GB:g}; use int8")
    return f"VRAM {vram:.0f} GB, precision {precision}"


def check_memory(summary):
    need = ram_bytes(summary)
    limit = memory_limit_bytes()
    budget = IMAGES_RAM_SHARE * limit if limit else IMAGES_RAM_FALLBACK_GB * 1024 ** 3
    gb = lambda b: f"{b / 1024 ** 3:.1f} GB"
    info = f"images in RAM {gb(need)}, container limit {gb(limit) if limit else 'unknown'}, budget {gb(budget)}"
    if need > budget:
        raise ValueError(f"generation too heavy for the worker ({info}): lower resolution/fps or max_length")
    return info


def _to_input(path, tag):
    """Copie une entree dans input/ de ComfyUI (LoadImage n'y lit que la)."""
    name = f"{tag}_{os.path.basename(path)}"
    shutil.copy(path, os.path.join(COMFY_DIR, "input", name))
    return name


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    prompt, negative = (params.get("prompt") or "").strip(), (params.get("negative") or "").strip()
    precision = params.get("precision") or "int8"
    work = tempfile.mkdtemp(prefix="wananimate_")
    copied = []
    tag = uuid.uuid4().hex[:8]
    prefix = f"fakinflu/{inp.get('run_id') or job.get('id', 'run')}"
    t0 = time.time()
    yield version_event("wananimate")
    if params.get("probe"):
        try:
            outputs = Outputs(inp)
            wait_comfy(120)
            report = probe_report()
            print(f"[wananimate] probe: {json.dumps(report)}", flush=True)
            path = os.path.join(work, "probe.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            yield from outputs.emit("probe", path, "probe.json")
            yield {"type": "done", "seed": None, "timings": {"total_s": round(time.time() - t0, 1)}}
        except Exception as e:
            traceback.print_exc()
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return
    try:
        # Signal et roles verifies AVANT tout telechargement ou calcul GPU.
        signal = signal_of(params)
        roles = required_inputs(params)
        missing = [r for r in roles if r not in (inp.get("inputs") or {})]
        if missing:
            raise ValueError(f"missing inputs for signal {signal}: {missing}")
        outputs = Outputs(inp)
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        missing = [r for r in roles if r not in paths]
        if missing:
            raise ValueError(f"missing inputs for signal {signal}: {missing}")
        video = "source" if signal == "pose" else "control"
        meta = ffprobe(paths[video])
        summary = plan(params, meta)
        mem_info = check_memory(summary)
        print(f"[wananimate] {mem_info} ; COMFY_EXTRA_ARGS={os.environ.get('COMFY_EXTRA_ARGS', '')!r}", flush=True)
        yield {"type": "progress", "step": mem_info, "pct": 1}
        yield {"type": "progress", "pct": 2,
               "step": f"{video} {meta['width']}x{meta['height']} {meta['fps']:.2f} fps, {meta['frames']} frames -> "
                       f"{summary['width']}x{summary['height']} at {summary['fps']:g} fps, {summary['frames_out']} frames"}

        files = {video: paths[video], "ref": _to_input(paths["ref"], tag)}
        if "mask" in paths:
            files["mask"] = paths["mask"]
        copied = [os.path.join(COMFY_DIR, "input", files["ref"])]
        wait_comfy()
        vram_info = check_vram(precision)
        if vram_info:
            yield {"type": "progress", "step": vram_info, "pct": 3}
        wf = build_workflow(params, prompt, negative, files, prefix, meta)
        seconds = yield from run_workflow(wf)

        produced = collect(prefix)
        if "video" not in produced:
            raise RuntimeError(f"no video produced (files: {sorted(produced)})")
        yield preview_event(thumb(produced["video"]), "image/jpeg")
        yield from outputs.emit("video", produced["video"], OUTPUT_NAMES["video"])
        outputs.write_summary()
        yield {"type": "done", "seed": params.get("seed"), "signal": signal,
               "timings": {"comfy_s": round(seconds, 1), "total_s": round(time.time() - t0, 1), "frames_out": summary["frames_out"]}}
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        for p in copied:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})

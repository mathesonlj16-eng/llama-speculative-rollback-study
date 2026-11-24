import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

if len(sys.argv) not in (6, 7):
    raise SystemExit("usage: profile32776.py OUTPUT SERVER TARGET DRAFT NSYS [NVTX_SHIM]")
out, binary, target, draft, nsys = [str(Path(p).resolve()) for p in sys.argv[1:6]]
out = Path(out)
out.mkdir()
port = 18091
instrument = Path(sys.argv[6]).resolve() if len(sys.argv) == 7 else None
url = f"http://127.0.0.1:{port}"
payload = dict(messages=[dict(role="user", content="Write 50 numbered, distinct facts about computer science. Give each fact one complete sentence. Do not stop early.")], temperature=0, max_tokens=160, chat_template_kwargs=dict(enable_thinking=False))

def request(stream=False):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(dict(payload, stream=stream)).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as response:
        if not stream:
            return json.load(response)
        events = []
        for line in response:
            stamp = time.monotonic_ns()
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                events.append(dict(monotonic_ns=stamp, event=json.loads(line[6:])))
        return events

def cli(*args):
    subprocess.run([nsys, *args], check=True, timeout=45)

def profile(mode):
    session = f"decode32776-{mode}-{os.getpid()}"
    server = [binary, "-m", target, "-ngl", "99", "-c", "1024", "-np", "1", "--jinja", "--host", "127.0.0.1", "--port", str(port)]
    if mode == "spec":
        server += ["-md", draft, "--spec-type", "draft-simple", "--spec-draft-n-max", "3", "-ngld", "99", "-td", "4"]
    (out / f"{mode}-command.json").write_text(json.dumps(server, indent=2))
    with (out / f"{mode}.log").open("w") as log:
        extra = ["--env-var=LD_PRELOAD=" + str(instrument)] if instrument else []
        launch = subprocess.Popen([nsys, "launch", "--session-new=" + session, "--trace=" + ("cuda,nvtx" if instrument else "cuda"), "--cuda-graph-trace=node", *extra, *server], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    if launch.poll() not in (None, 0):
                        raise RuntimeError(f"launch exited {launch.returncode}; see {log.name}")
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"server did not become ready; see {log.name}")
            warm = request()
            controls = [request() for _ in range(1)]
            control = controls[0]
            gpu_memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip()
            (out / f"{mode}-control.json").write_text(json.dumps(dict(controls=controls, gpu_memory=gpu_memory, instrument=str(instrument)), indent=2))
            cli("start", "--session=" + session, "--sample=none", "--cpuctxsw=none", "--output=" + str(out / mode))
            before = time.monotonic_ns()
            events = request(stream=True)
            after = time.monotonic_ns()
            (out / f"{mode}-requests.json").write_text(json.dumps(dict(warm=warm, control=control, before_ns=before, after_ns=after, events=events), indent=2))
            cli("stop", "--session=" + session)
            cli("export", "--type=sqlite", "--output=" + str(out / f"{mode}.sqlite"), str(out / f"{mode}.nsys-rep"))
        finally:
            subprocess.run([nsys, "shutdown", "--session=" + session, "--kill=sigterm"], timeout=20)
            if launch.poll() is None:
                launch.terminate()
                launch.wait(timeout=10)

print(out, flush=True)
os.environ.pop("LLAMA32776_TARGET_ROLLBACK", None)
for mode in (("spec",) if instrument else ("baseline", "spec")):
    profile(mode)

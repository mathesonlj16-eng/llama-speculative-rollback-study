import json
import statistics
import sys
import urllib.request
from pathlib import Path

mode = sys.argv[1]
url = "http://127.0.0.1:8091/v1/chat/completions"

payload = {
    "messages": [{
        "role": "user",
        "content": (
            "Write 50 numbered, distinct facts about computer science. "
            "Give each fact one complete sentence. Do not stop early."
        )
    }],
    "temperature": 0,
    "max_tokens": 160,
    "chat_template_kwargs": {"enable_thinking": False}
}

results = []

for i in range(6):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)

    t = data["timings"]

    if i == 0:
        print("Warm-up complete")
        continue

    results.append(t)
    print(
        f"Run {i}: "
        f"{t['predicted_per_second']:.2f} tok/s, "
        f"{t['predicted_n']} tokens, "
        f"accepted {t.get('draft_n_accepted', 'N/A')}/"
        f"{t.get('draft_n', 'N/A')}"
    )

speeds = [r["predicted_per_second"] for r in results]
print(f"\n{mode} median: {statistics.median(speeds):.2f} tok/s")

path = Path(f"bench_{mode}.json")
with path.open("x") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {path}")

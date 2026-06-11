"""Quick end-to-end smoke test against a running server (not part of pytest)."""
import json
import sys
import urllib.request
import urllib.error

sys.stdout.reconfigure(errors="replace")  # legacy Windows consoles aren't UTF-8

BASE = "http://localhost:7860"


def post(path, body=None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req)), 200
    except urllib.error.HTTPError as e:
        return json.load(e), e.code


obs, _ = post("/reset?task_id=debt_trap&seed=42&session_id=e2e")
print("reset:", obs["net_worth"], obs["market_regime"])

for m in range(6):
    r, code = post("/step?session_id=e2e", {"action_type": "pay_credit_card", "amount": 8000})
    i = r["info"]
    print(f"month {m+1}: reward={r['reward']:.2f} done={r['done']} event={i['event']} "
          f"ok={i['action_ok']} cash={i['cash_flow']:.0f} net={i['net_worth']:.0f}")

g, _ = post("/grade?session_id=e2e")
print("grade:", round(g["score"], 3), [c["label"] for c in g["components"]])

r, code = post("/step?session_id=e2e", {"action_type": "fly_to_moon"})
print("bad action ->", code, str(r["detail"])[:70])
r, code = post("/step?session_id=ghost", {"action_type": "hold"})
print("no session ->", code, str(r["detail"])[:70])
r, code = post("/reset?task_id=bogus&session_id=e2e")
print("bad task ->", code, str(r["detail"])[:70])

html = urllib.request.urlopen(BASE + "/").read().decode()
print("frontend served:", "FinAgent" in html and "chart.js" in html.lower())
for asset in ("/app.js", "/style.css"):
    status = urllib.request.urlopen(BASE + asset).status
    print(f"asset {asset}: {status}")

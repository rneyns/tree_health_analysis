import json, time, requests, jwt  # pip install pyjwt cryptography

import json, sys, re
from pathlib import Path

p = Path('/Users/robbe_neyns/Downloads/service_key.json')  # <-- your file

raw = p.read_text(encoding="utf-8-sig")  # tolerates BOM
print("HEAD repr:", repr(raw[:80]))      # show hidden chars

raw_stripped = raw.strip()

# If the whole thing is accidentally wrapped in quotes/backticks, un-wrap it
if (raw_stripped.startswith(("'", '"', "`"))
    and raw_stripped.endswith(("'", '"', "`"))
    and raw_stripped[1] == "{"):
    raw = raw_stripped.strip("'").strip('"').strip("`")

# If the private_key contains literal newlines, JSON will choke — escape them
if "-----BEGIN" in raw and "\n" in raw:
    # Ensure the value is a single JSON string with \n escapes
    raw = re.sub(
        r'("private_key"\s*:\s*")([\s\S]*?)(")',
        lambda m: m.group(1)
                  + m.group(2).replace("\\", "\\\\").replace("\r\n","\n").replace("\n", r"\n")
                  + m.group(3),
        raw,
        count=1
    )

try:
    sk = json.loads(raw)
except json.JSONDecodeError as e:
    print("\nJSON error:", e)
    # Show offending line to spot the issue
    lines = raw.splitlines()
    ln = getattr(e, "lineno", 1) or 1
    print("Offending line repr:", repr(lines[ln-1] if ln-1 < len(lines) else ""))
    sys.exit(1)

print("✅ Loaded JSON keys:", list(sk.keys()))

# ---- 0) Load your service key properly (JSON!) ----
# If it's in a file that contains *valid JSON* (double quotes), do:
with open('/Users/robbe_neyns/Downloads/service_key.json', "r", encoding="utf-8") as f:
    sk = json.load(f)

# If you truly have it as a Python string variable called service_key_str,
# make sure it is valid JSON (double quotes) then:
# sk = json.loads(service_key_str)

client_id   = sk["client_id"]
user_id     = sk["user_id"]
token_uri   = sk["token_uri"]
private_key = sk["private_key"]

# ---- 1) Build a short-lived JWT (RS256) ----
now = int(time.time())
claims = {
    "iss": client_id,
    "sub": user_id,
    "aud": token_uri,
    "iat": now,
    "exp": now + 5*60,
}
assertion = jwt.encode(claims, private_key, algorithm="RS256")
# pyjwt>=2 returns str; older may return bytes:
if isinstance(assertion, bytes):
    assertion = assertion.decode("utf-8")

# ---- 2) Exchange JWT for access token ----
tok_resp = requests.post(
    token_uri,
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion
    },
    headers={
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    },
)

# Safely decode JSON (or show the real server error text)
if "application/json" not in (tok_resp.headers.get("Content-Type","")):
    print("Token endpoint did not return JSON. Status:", tok_resp.status_code)
    print(tok_resp.text[:1000])  # <- this will show the HTML/text error page
    raise SystemExit

tok = tok_resp.json()             # now it's safe
access_token = tok["access_token"]

# ---- 3) Example authorized call (search) ----
resp = requests.get(
    "https://land.copernicus.eu/api/@search",
    params={"portal_type":"DataSet","b_size":10},
    headers={"Accept":"application/json","Authorization": f"Bearer {access_token}"},
)

if "application/json" not in (resp.headers.get("Content-Type","")):
    print("Search did not return JSON. Status:", resp.status_code)
    print(resp.text[:1000])
    raise SystemExit

print(resp.json())

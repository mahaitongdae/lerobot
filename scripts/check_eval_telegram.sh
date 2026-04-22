#!/usr/bin/env bash
# Check SSL sweep eval progress and send results via Telegram.
#
# Usage:
#   bash scripts/check_eval_telegram.sh [--results-dir DIR] [--bot-token TOKEN] [--chat-id ID]
#
# Environment variables (CLI flags override):
#   RESULTS_DIR     Directory containing SSL sweep results (default: results/ssl_sweep_allsuites)
#   TELEGRAM_TOKEN  Bot API token
#   TELEGRAM_CHAT   Chat ID to send to
#   SSH_HOST        Remote host to check running processes (optional, e.g. haitongma@10.243.49.216)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
RESULTS_DIR="${RESULTS_DIR:-results/ssl_sweep_allsuites}"
TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-8414933122:AAEMghwrCAAzmlhCjIKpu-HQdzxGH-lou5I}"
TELEGRAM_CHAT="${TELEGRAM_CHAT:-8748647942}"
SSH_HOST="${SSH_HOST:-}"

# ── Parse CLI flags ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --bot-token)   TELEGRAM_TOKEN="$2"; shift 2 ;;
        --chat-id)     TELEGRAM_CHAT="$2"; shift 2 ;;
        --ssh-host)    SSH_HOST="$2"; shift 2 ;;
        -h|--help)     sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

RESULTS_DIR="${RESULTS_DIR%/}"

# ── Count progress ───────────────────────────────────────────────
COMPLETED=$(find "${RESULTS_DIR}"/SSL_*/checkpoints/*/eval -name eval_info.json 2>/dev/null | grep -v /last/ | wc -l)
TOTAL=$(find "${RESULTS_DIR}"/SSL_*/checkpoints/*/pretrained_model -maxdepth 0 -type d 2>/dev/null | grep -v /last/ | wc -l)
PCT=$(python3 -c "print(f'{${COMPLETED}/${TOTAL}*100:.1f}' if ${TOTAL}>0 else '0.0')")

# ── Check running processes ───────────────────────────────────────
RUNNING_INFO=""
if [[ -n "$SSH_HOST" ]]; then
    RUNNING_INFO=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$SSH_HOST" \
        "ps -eo etime,cmd | grep lerobot-eval | grep -v grep" 2>/dev/null || true)
fi
# Fallback: check locally
if [[ -z "$RUNNING_INFO" ]]; then
    RUNNING_INFO=$(ps -eo etime,cmd | grep lerobot-eval | grep -v grep 2>/dev/null || true)
fi

NUM_RUNNING=$(echo "$RUNNING_INFO" | grep -c lerobot-eval 2>/dev/null || echo 0)

# Extract current checkpoint from first running process
CURRENT_CKPT=""
if [[ "$NUM_RUNNING" -gt 0 ]]; then
    CURRENT_CKPT=$(echo "$RUNNING_INFO" | head -1 | grep -oP 'SSL_[^/]+' | head -1 || true)
fi

# ── Extract results ───────────────────────────────────────────────
RESULTS_TABLE=$(python3 << 'PYEOF'
import json, os, sys
from collections import defaultdict

results_dir = os.environ.get("RESULTS_DIR", "results/ssl_sweep_allsuites")

# Collect all eval results
data = []  # (run, step, success_rate)
for root, dirs, files in os.walk(results_dir):
    if "eval_info.json" in files and "/eval" in root:
        # Skip .cache directories
        if "/.cache/" in root:
            continue
        # Skip /last/ checkpoints
        if "/last/" in root:
            continue
        f = os.path.join(root, "eval_info.json")
        try:
            d = json.load(open(f))
            sr = d["overall"]["pc_success"]
            # Parse run name and step
            rel = os.path.relpath(root, results_dir)
            parts = rel.split("/")
            run = parts[0]
            step = parts[2] if len(parts) > 2 else "?"
            data.append((run, step, sr))
        except Exception:
            pass

if not data:
    print("No results yet.")
    sys.exit(0)

# Parse run name into components
def parse_run(run):
    # SSL_act_libero_10_byol -> policy=act, suite=libero_10, backbone=byol
    parts = run.split("_")
    policy = parts[1]  # act or dp
    # Find suite (libero_10, libero_goal, libero_object, libero_spatial)
    suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    suite = ""
    backbone = ""
    rest = "_".join(parts[2:])
    for s in suites:
        if rest.startswith(s):
            suite = s
            backbone = rest[len(s)+1:]
            break
    return policy, suite, backbone

# Group by policy x suite x backbone, get best (100K or max step)
best_results = {}  # (policy, suite, backbone) -> (step, sr)
all_steps = defaultdict(list)  # (policy, suite, backbone) -> [(step, sr)]

for run, step, sr in data:
    policy, suite, backbone = parse_run(run)
    key = (policy, suite, backbone)
    all_steps[key].append((step, sr))
    if step == "100000":
        best_results[key] = (step, sr)
    elif key not in best_results:
        if not best_results.get(key) or sr > best_results[key][1]:
            best_results[key] = (step, sr)

# Print summary by suite
suites_seen = sorted(set(s for _, s, _ in best_results.keys()))
policies_seen = sorted(set(p for p, _, _ in best_results.keys()))
backbones_seen = sorted(set(b for _, _, b in best_results.keys()))

print("<b>Best results by backbone (@ 100K or latest):</b>")
print()

for suite in suites_seen:
    print(f"📋 <b>{suite}</b>")
    rows = []
    for backbone in backbones_seen:
        for policy in policies_seen:
            key = (policy, suite, backbone)
            if key in best_results:
                step, sr = best_results[key]
                step_k = f"{int(step)//1000}K" if step.isdigit() else step
                rows.append((sr, f"  {policy.upper()} {backbone}: {sr:.1f}% ({step_k})"))
    rows.sort(key=lambda x: -x[0])
    for _, line in rows:
        print(line)
    print()

# Print summary table for 100K results
results_100k = [(p, s, b, sr) for (p, s, b), (step, sr) in best_results.items() if step == "100000"]
if results_100k:
    print("<b>Summary table @ 100K:</b>")
    print("<pre>")
    header = f"{'Backbone':<12}"
    for suite in suites_seen:
        short = suite.replace("libero_", "")
        header += f" {short:>8}"
    print(header)
    print("-" * len(header))
    for backbone in backbones_seen:
        for policy in policies_seen:
            line = f"{policy.upper()}/{backbone:<7}"
            has_data = False
            for suite in suites_seen:
                key = (policy, suite, backbone)
                if key in best_results and best_results[key][0] == "100000":
                    line += f" {best_results[key][1]:>7.1f}%"
                    has_data = True
                else:
                    line += f" {'—':>8}"
            if has_data:
                print(line)
    print("</pre>")
PYEOF
)

# ── Build message ─────────────────────────────────────────────────
MSG="📊 <b>SSL Sweep Eval Report</b>

<b>Progress:</b> ${COMPLETED}/${TOTAL} checkpoints (${PCT}%)
<b>Running:</b> ${NUM_RUNNING} processes"

if [[ -n "$CURRENT_CKPT" ]]; then
    MSG="${MSG}
<b>Current:</b> <code>${CURRENT_CKPT}</code>"
fi

MSG="${MSG}

${RESULTS_TABLE}"

# ── Check if all done ─────────────────────────────────────────────
if [[ "$COMPLETED" -eq "$TOTAL" ]] && [[ "$TOTAL" -gt 0 ]]; then
    MSG="${MSG}

✅ <b>All ${TOTAL} evaluations complete!</b>"
fi

# ── Send via Telegram ─────────────────────────────────────────────
RESPONSE=$(curl -s "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT}" \
    -d parse_mode=HTML \
    -d text="${MSG}")

# Check if send was successful
OK=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok', False))" 2>/dev/null || echo "False")
if [[ "$OK" == "True" ]]; then
    echo "Telegram message sent successfully (${COMPLETED}/${TOTAL} complete)"
else
    echo "Failed to send Telegram message:" >&2
    echo "$RESPONSE" >&2
    # Print to stdout as fallback
    echo ""
    echo "$MSG"
    exit 1
fi

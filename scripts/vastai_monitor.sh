#!/usr/bin/env bash
set -euo pipefail

TEAM_API_KEY="f31d90635074ab7c55d335e87c3c690b26b9c3a98b681f25b527d70f63ec184e"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"

declare -A MACHINES=(
    ["35942000|8x RTX 4070S Ti|team"]="ssh6.vast.ai:22000:1.87"
    ["35942892|8x RTX 4090|team"]="ssh4.vast.ai:22892:2.32"
    ["35860059|8x RTX 5090|personal"]="ssh2.vast.ai:20058:5.18"
)

timestamp=$(date '+%Y-%m-%d %H:%M')
report="Vast.ai GPU Monitor - ${timestamp}\n"
team_cost=0
personal_cost=0

for key in "${!MACHINES[@]}"; do
    IFS='|' read -r id model account <<< "$key"
    IFS=':' read -r host port cost_hr <<< "${MACHINES[$key]}"

    report+="\n[$id] ${model} (${account}) - \$${cost_hr}/hr\n"

    gpu_info=$(ssh $SSH_OPTS -p "$port" "root@${host}" \
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>&1) || gpu_info="SSH FAILED"

    if [[ "$gpu_info" == "SSH FAILED" ]]; then
        report+="  !! SSH connection failed\n"
    else
        avg_util=0
        total_mem_used=0
        total_mem=0
        gpu_count=0
        while IFS=',' read -r idx util mem_used mem_total temp; do
            util=$(echo "$util" | tr -d ' %')
            mem_used=$(echo "$mem_used" | tr -d ' MiB')
            mem_total=$(echo "$mem_total" | tr -d ' MiB')
            temp=$(echo "$temp" | tr -d ' ')
            report+="  GPU${idx}: ${util}% util | ${mem_used}/${mem_total} MiB | ${temp}C\n"
            avg_util=$((avg_util + util))
            total_mem_used=$((total_mem_used + mem_used))
            total_mem=$((total_mem + mem_total))
            gpu_count=$((gpu_count + 1))
        done <<< "$gpu_info"
        if [[ $gpu_count -gt 0 ]]; then
            avg_util=$((avg_util / gpu_count))
            report+="  Avg util: ${avg_util}% | Mem: ${total_mem_used}/${total_mem} MiB\n"
        fi
    fi

    top_procs=$(ssh $SSH_OPTS -p "$port" "root@${host}" \
        "ps aux --sort=-%cpu | grep -v '^\(root.*sshd\|root.*bash\|USER\|root.*ps\|root.*grep\|root.*sort\)' | head -5 | awk '{printf \"  %-6s %5s%% CPU  %s\n\", \$2, \$3, substr(\$0, index(\$0,\$11))}'" 2>&1) || top_procs=""

    if [[ -n "$top_procs" ]]; then
        report+="  Top processes:\n${top_procs}\n"
    fi

    if [[ "$account" == "team" ]]; then
        team_cost=$(echo "$team_cost + $cost_hr" | bc)
    else
        personal_cost=$(echo "$personal_cost + $cost_hr" | bc)
    fi
done

team_credit=$(vastai show invoices --api-key "$TEAM_API_KEY" 2>/dev/null \
    | grep -o "'credit': [0-9.]*" | grep -o "[0-9.]*" | head -1 \
    | xargs printf "%.2f" 2>/dev/null) || team_credit="N/A"
personal_credit=$(vastai show invoices 2>/dev/null \
    | grep -o "'credit': [0-9.]*" | grep -o "[0-9.]*" | head -1 \
    | xargs printf "%.2f" 2>/dev/null) || personal_credit="N/A"

report+="\n--- SUMMARY ---\n"
report+="Team (cong-harvard): \$${team_cost}/hr | Credit: \$${team_credit}"
if [[ "$team_credit" != "N/A" ]]; then
    team_hrs=$(echo "scale=1; $team_credit / $team_cost" | bc)
    report+=" | ~${team_hrs}h left"
fi
report+="\n"
report+="Personal: \$${personal_cost}/hr | Credit: \$${personal_credit}"
if [[ "$personal_credit" != "N/A" ]]; then
    personal_hrs=$(echo "scale=1; $personal_credit / $personal_cost" | bc)
    report+=" | ~${personal_hrs}h left"
fi
report+="\n"

echo -e "$report"

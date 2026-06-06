#!/bin/bash
# Score every b-roll clip with Higgsfield's Virality Predictor and rank them.
# Tells you which clip / reel concept is predicted to perform best.
set -a; . ~/.config/higgsfield/credentials.env 2>/dev/null; set +a
cd /Users/samarthvaka/20-watts || exit 1
OUT=assets/video
CSV=results/virality_scores.csv
echo "clip,overall,viral_potential,hook,sustain,brain_engagement" > "$CSV"
for f in "$OUT"/*.mp4; do
  name=$(basename "$f" .mp4)
  echo "[score] $name ..."
  higgsfield generate create brain_activity --video "$f" --wait --wait-timeout 6m --json \
    > "$OUT/${name}_viral.json" 2>/dev/null
  python3 - "$name" "$OUT/${name}_viral.json" >> "$CSV" <<'PY'
import json, sys
name, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
    s = d[0]["params"]["analysis"]["scores"]
    print(f'{name},{s.get("overall_score")},{s.get("viral_potential")},{s.get("hook_score")},{s.get("sustain")},{s.get("brain_engagement")}')
except Exception as e:
    print(f'{name},ERR,{e},,,')
PY
done
echo ""
echo "=== RANKED by viral_potential (Higgsfield Virality Predictor) ==="
{ head -1 "$CSV"; tail -n +2 "$CSV" | sort -t, -k3 -rn; } | column -t -s,

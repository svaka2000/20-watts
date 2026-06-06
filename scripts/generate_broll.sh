#!/bin/bash
# Animate the 20 Watts hero stills into vertical (9:16) cinematic b-roll clips for the reels.
# Usage: bash scripts/generate_broll.sh [MODEL] [DURATION]
set -a; . ~/.config/higgsfield/credentials.env 2>/dev/null; set +a
cd /Users/samarthvaka/20-watts || exit 1
MODEL="${1:-veo3_1_lite}"
DUR="${2:-4}"
OUT=assets/video
mkdir -p "$OUT"

clip() {
  local name="$1" img="$2" prompt="$3"
  if [ -f "$OUT/$name.mp4" ]; then echo "[skip] $name exists"; return; fi
  echo "[gen] $name ($MODEL ${DUR}s) ..."
  higgsfield generate create "$MODEL" --prompt "$prompt" --aspect_ratio 9:16 --duration "$DUR" \
    --image "$img" --wait --wait-timeout 12m --json > "$OUT/$name.json" 2>"$OUT/$name.err"
  local url
  url=$(python3 -c "import json,re;print((re.findall(r'https?://[^\"]+\.mp4[^\"]*', json.dumps(json.load(open('$OUT/$name.json')))) or [''])[0])" 2>/dev/null)
  if [ -n "$url" ]; then curl -sL "$url" -o "$OUT/$name.mp4" && echo "[ok] $name -> $OUT/$name.mp4"; else echo "[fail] $name"; head -2 "$OUT/$name.err"; fi
}

clip ep1_ai_dense assets/ep1_ai_dense.png "Dense artificial neural network where every single node blazes with hot white light at once, overwhelming and energy-wasteful, subtle pulsing and heat shimmer, cinematic, dark background"
clip series_20watts assets/series_20watts.png "A warm glowing lightbulb filament beside a towering cold blue server rack, the filament gently flickering and the server LEDs blinking, dramatic scale contrast, slow cinematic push-in, dark studio"
clip ep2_depth assets/ep2_depth.png "Tall tower of glowing translucent horizontal layers, a beam of light entering at the bottom and exiting early through a side opening halfway up while the upper layers stay dark and unused, ethereal, slow camera tilt"
clip ep3_attention_sink assets/ep3_attention_sink.png "A single brilliant anchor point of light at the start of a long dark flowing stream, pulling in ribbons of light like a pressure release valve, slow motion, cinematic blue and gold"
clip ep3_memory_fade assets/ep3_memory_fade.png "A long horizontal ribbon of glowing text, the recent words on the right staying bright while the middle dissolves into drifting dark particles, ethereal slow drift, cinematic"
clip ep4_synaptic_pruning assets/ep4_synaptic_pruning.png "A glowing neural network being pruned, excess synaptic connections trimming and dissolving away like branches while the remaining connections grow brighter and cleaner, slow cinematic motion, blue and gold, dark background"

echo "DONE broll"; ls -la "$OUT"/*.mp4 2>/dev/null | awk '{print $5, $9}'

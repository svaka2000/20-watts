#!/bin/bash
# Generate cinematic reel visuals for the 20 Watts series via Higgsfield CLI.
# Uses subscription credits. ~0.15 credits / image. Vertical 9:16 for reels.
set -a; . ~/.config/higgsfield/credentials.env 2>/dev/null; set +a
OUT=/Users/samarthvaka/20-watts/assets
mkdir -p "$OUT"

gen() {
  local name="$1"; local prompt="$2"; local ar="${3:-9:16}"
  echo "[gen] $name ..."
  higgsfield generate create soul_cinematic --prompt "$prompt" \
    --aspect_ratio "$ar" --quality 2k --wait --json > "$OUT/$name.json" 2>"$OUT/$name.err"
  local url
  url=$(python3 -c "import json;d=json.load(open('$OUT/$name.json'));print((d[0].get('result_url') if isinstance(d,list) else d.get('result_url')) or '')" 2>/dev/null)
  if [ -n "$url" ]; then
    curl -sL "$url" -o "$OUT/$name.png" && echo "[ok] $name -> $OUT/$name.png"
  else
    echo "[fail] $name (see $name.err)"; head -3 "$OUT/$name.err" 2>/dev/null
  fi
}

gen ep1_brain_sparse "Cinematic macro photograph of a glowing human brain floating in darkness, only a tiny scattered handful of neurons lit bright electric blue while the vast majority stay dark and dormant, dramatic rim lighting, scientific awe, ultra detailed, deep black background, conceptual energy efficiency"
gen ep1_ai_dense "Cinematic visualization of an artificial neural network as a dense glowing grid where every single node blazes with hot white light at once, overwhelming and energy wasteful, contrasted against darkness, futuristic, ultra detailed"
gen series_20watts "A human brain glowing softly like a warm 20 watt lightbulb filament beside a towering server rack blazing with cold blue light, dramatic scale contrast, cinematic conceptual, dark moody studio background, energy comparison"
gen ep2_depth "Cinematic conceptual image of a tall tower of glowing translucent horizontal layers, a beam of light entering at the bottom and exiting early through a side door halfway up, the upper layers dark and unused, ethereal futuristic"
gen ep3_memory_fade "Cinematic conceptual visualization of memory fading, a long horizontal ribbon of glowing text where only the most recent words on the right and the very first word on the left stay bright while the middle dissolves into drifting dark particles, ethereal dramatic"
gen ep3_attention_sink "A single brilliant anchor point of light at the start of a long dark flowing stream, pulling in ribbons of light like a pressure release valve, conceptual attention sink, cinematic dramatic blue and gold, deep dark background"
gen series_hero_thumbnail "Bold cinematic thumbnail, a human brain made of circuit traces glowing at low warm power on the left versus a hot overloaded GPU on the right, lightning of energy between them, dramatic high contrast, text space at top, viral science aesthetic" 16:9

echo "DONE visuals"
ls -la "$OUT"/*.png 2>/dev/null | awk '{print $5, $9}'

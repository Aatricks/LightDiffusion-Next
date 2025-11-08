#!/usr/bin/env bash
set -euo pipefail

REPO="Aatricks/LightDiffusion-Next"
TOKEN="${GITHUB_TOKEN}"

BADGE_PATH=".github/badges/views.svg"
DATA_PATH="views.json"

json=$(curl -s -H "Authorization: Bearer $TOKEN" \
              -H "Accept: application/vnd.github+json" \
              "https://api.github.com/repos/$REPO/traffic/views")

views_14d=$(echo "$json" | grep -o '"count":[0-9]*' | cut -d: -f2 | paste -sd+ - | bc)
views_14d=${views_14d:-0}

if [ ! -f "$DATA_PATH" ]; then
    echo '{"history":[]}' > "$DATA_PATH"
fi

date=$(date -u +"%Y-%m-%d")
tmp=$(mktemp)
jq --arg d "$date" --argjson c "$views_14d" \
   '.history += [{date:$d, views:$c}]' \
   "$DATA_PATH" > "$tmp"
mv "$tmp" "$DATA_PATH"

total=$(jq '[.history[].views] | add // 0' "$DATA_PATH")

mkdir -p "$(dirname "$BADGE_PATH")"

cat > "$BADGE_PATH" <<EOF
<svg xmlns="http://www.w3.org/2000/svg" width="160" height="20">
<linearGradient id="b" x2="0" y2="100%">
  <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
  <stop offset="1" stop-opacity=".1"/>
</linearGradient>
<mask id="a">
  <rect width="160" height="20" rx="3" fill="#fff"/>
</mask>
<g mask="url(#a)">
  <rect width="110" height="20" fill="#555"/>
  <rect x="110" width="50" height="20" fill="#007ec6"/>
  <rect width="160" height="20" fill="url(#b)"/>
</g>
<g fill="#fff" text-anchor="middle"
   font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
  <text x="55" y="15">views</text>
  <text x="135" y="15">$total</text>
</g>
</svg>
EOF

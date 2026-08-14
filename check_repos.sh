#!/bin/bash
# Check GitHub repos for all dependencies
for repo in "Aratako/MioCodec" "Datadog/Toto" "google-deepmind/videoprism" "IBM/tsfm" "NeoQuasar/Kronos" "YuchenShen/FoMo"; do
  echo "=== $repo ==="
  curl -s --connect-timeout 10 --max-time 20 "https://api.github.com/repos/$repo" 2>&1 | grep -E '"full_name"|"default_branch"|"message"' | head -3
  echo ""
done

#!/bin/bash
for model in aratako-miocodec-25hz-44-1khz-v2 datadog-toto-2-0-22m google-videoprism-lvt-base-f16r288 ibm-granite-granite-timeseries-patchtst-fm-r1 ibm-research-ttm-r3 neoquasar-kronos-base openmoss-team-moss-voicegenerator yuchenshen-fomo-0d; do
  echo "========================================"
  echo "=== $model ==="
  echo "========================================"
  cat /workspace/models/$model/create_env.sh 2>/dev/null
  echo ""
done

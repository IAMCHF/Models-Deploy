#!/bin/bash
echo "=== IBM/tsfm redirect ==="
curl -s -I --connect-timeout 10 --max-time 20 "https://api.github.com/repos/IBM/tsfm" 2>&1 | grep -iE "location|HTTP" | head -5

echo ""
echo "=== Search Kronos ==="
curl -s --connect-timeout 10 --max-time 20 "https://api.github.com/search/repositories?q=Kronos+timeseries+foundation+model&per_page=5" 2>&1 | grep '"full_name"' | head -5

echo ""
echo "=== Search FoMo ==="
curl -s --connect-timeout 10 --max-time 20 "https://api.github.com/search/repositories?q=FoMo+0D&per_page=5" 2>&1 | grep '"full_name"' | head -5

echo ""
echo "=== Search toto2 ==="
curl -s --connect-timeout 10 --max-time 20 "https://api.github.com/search/repositories?q=toto+timeseries+foundation+model&per_page=5" 2>&1 | grep '"full_name"' | head -5

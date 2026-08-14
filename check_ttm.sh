#!/bin/bash
echo "=== ibm-research-ttm-r3 startup log ==="
cat /tmp/ibm-research-ttm-r3_test.log 2>/dev/null | tail -30
echo ""
echo "=== ibm-research-ttm-r3 test result ==="
cat /tmp/ibm-research-ttm-r3_test_result.log 2>/dev/null | tail -20
echo ""
echo "=== check weights dir ==="
ls /workspace/models/ibm-research-ttm-r3/weights/ 2>/dev/null | head -20

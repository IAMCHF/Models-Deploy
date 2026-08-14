# ============================================================
# resume_run_all.ps1 - 断电后续跑剩余 15 个模型
# 前 24 个结果见 result-0815.txt (03:24 轮)
# 逻辑与 run_all_host_test.ps1 完全一致: --network none + bash start.sh
# ============================================================

$ErrorActionPreference = "Continue"

$ROOT   = "d:\ssd-projects\Models-Deploy"
$RESULT = "$ROOT\result-0815.txt"
$IMAGE  = "models-deploy-base:latest"

$MODELS = [ordered]@{
  "openmoss-team-moss-tts-local-transformer-v1-5" = 139
  "openmoss-team-moss-voicegenerator"             = 139
  "paddlepaddle-pp-chart2table"                   = 25
  "paddlepaddle-pp-docblocklayout"                = 23
  "paddlepaddle-pp-doclayout-plus-l"              = 25
  "paddlepaddle-pp-ocrv6-medium-det-onnx"         = 60
  "paddlepaddle-pp-ocrv6-small-det-onnx"          = 60
  "paddlepaddle-pp-ocrv6-small-rec-onnx"          = 60
  "prior-labs-tabpfn-v2-clf"                      = 11
  "prior-labs-tabpfn-v2-reg"                      = 11
  "skywork-skywork-reward-v2-qwen3-0-6b"          = 27
  "synthefy-nori-30m"                             = 5
  "voyageai-voyage-4-nano"                        = 35
  "weborganizer-topicclassifier-nourl"            = 9
  "yuchenshen-fomo-0d"                            = 13
}

$passCount = 0
$failCount = 0
$failList  = @()

function Write-Result([string]$text) {
    $text | Out-File -FilePath $RESULT -Append -Encoding utf8
}

$HEALTH_PY = "import urllib.request,sys;`ntry:`n r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3);`n sys.exit(0 if b'ok' in r.read() else 1)`nexcept Exception:`n sys.exit(1)"

Write-Result "========================================"
Write-Result "Resume remaining 15 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (after power loss, offline, --network none)"
Write-Result "========================================"

foreach ($model in $MODELS.Keys) {
    $startup = $MODELS[$model]
    $maxWait = $startup * 2 + 60
    $mount   = "$ROOT\models\$model"

    Write-Host ""
    Write-Host ">>> [$model] start container (network=none, max wait ${maxWait}s)"

    docker rm -f $model 2>$null | Out-Null

    $runOut = docker run -d --name $model --gpus all --network none `
        -v "${mount}:/workspace/models/${model}" `
        -w "/workspace/models/${model}" `
        $IMAGE bash start.sh 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] docker run failed: $runOut"
        $failCount++
        $failList += $model
        Write-Result "[FAIL] $model | docker run failed: $runOut"
        continue
    }

    $ready   = $false
    $elapsed = 0
    while ($elapsed -lt $maxWait) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        docker exec $model python3 -c $HEALTH_PY 2>$null
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            break
        }
        $state = docker inspect -f '{{.State.Running}}' $model 2>$null
        if ($state -ne "true") {
            break
        }
    }

    if ($ready) {
        Write-Host "  [OK] service ready (${elapsed}s)"
        Write-Host ">>> [$model] run test.py (in container)"
        $testOut = docker exec $model python3 test/test.py 2>&1 | ForEach-Object { $_.ToString() }
        $testExit = $LASTEXITCODE
        if ($testExit -eq 0) {
            $passCount++
            Write-Host "  [PASS] $model"
            Write-Result "[PASS] $model | startup ${elapsed}s | test PASS"
        } else {
            $failCount++
            $failList += $model
            Write-Host "  [FAIL] $model test failed (exit=$testExit)"
            Write-Result "[FAIL] $model | startup ${elapsed}s | test FAIL (exit=$testExit)"
        }
        foreach ($line in $testOut) { Write-Result "    $line" }
    } else {
        $failCount++
        $failList += $model
        Write-Host "  [FAIL] $model startup timeout/exit"
        Write-Result "[FAIL] $model | startup timeout (>${maxWait}s) or process exited"
        $log = docker logs $model 2>&1 | Select-Object -Last 40
        foreach ($line in $log) { Write-Result "    $line" }
    }

    docker stop $model 2>$null | Out-Null
    docker rm  $model 2>$null | Out-Null
    Start-Sleep -Seconds 3
}

Write-Result "========================================"
Write-Result "Resume Summary: PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) {
    Write-Result "Resume Failed: $($failList -join ', ')"
}
Write-Result "========================================"
Write-Host ""
Write-Host "Done. PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"

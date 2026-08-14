# ============================================================
# retest_affected.ps1 - 复测 6 个受修复影响的模型
# 严格真实形态: docker run 单挂载 + bash start.sh, --network none
# 健康检查与 test.py 均通过 docker exec 在容器内执行
# 失败时抓取完整容器日志(诊断 videomaev2 JSON 解析错误)
# 结果写入 result-0815-retest.txt
# ============================================================

$ErrorActionPreference = "Continue"

$ROOT   = "d:\ssd-projects\Models-Deploy"
$RESULT = "$ROOT\result-0815-retest.txt"
$IMAGE  = "models-deploy-base:latest"

$MODELS = [ordered]@{
  "paddlepaddle-pp-ocrv6-small-det-onnx"  = 60
  "paddlepaddle-pp-ocrv6-medium-det-onnx" = 60
  "paddlepaddle-pp-ocrv6-small-rec-onnx"  = 60
  "yuchenshen-fomo-0d"                    = 30
  "numind-nuextract3-fp8"                 = 30
  "opengvlab-videomaev2-base"             = 60
}

$passCount = 0
$failCount = 0
$failList  = @()

function Write-Result([string]$text) {
    $text | Out-File -FilePath $RESULT -Append -Encoding utf8
}

$HEALTH_PY = "import urllib.request,sys;`ntry:`n r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3);`n sys.exit(0 if b'ok' in r.read() else 1)`nexcept Exception:`n sys.exit(1)"

Write-Result "========================================"
Write-Result "Affected-models retest - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (offline, --network none, via bash start.sh)"
Write-Result "Total: $($MODELS.Count) models"
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
Write-Result "Summary: PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) {
    Write-Result "Failed: $($failList -join ', ')"
}
Write-Result "========================================"
Write-Host ""
Write-Host "Done. PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) { Write-Host "Failed: $($failList -join ', ')" }
Write-Host "Result file: $RESULT"

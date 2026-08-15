# 重测剩余 6 个: mitra(脚本已修), koreapeter×2(pandas已装3.10), openmoss×2+docblocklayout(断电未跑)
$ErrorActionPreference = "Continue"
$ROOT   = "d:\ssd-projects\Models-Deploy"
$RESULT = "$ROOT\result-0815.txt"
$IMAGE  = "models-deploy-base:latest"

$MODELS = [ordered]@{
  "autogluon-mitra-classifier"                       = 30
  "koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus" = 25
  "koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus" = 25
  "openmoss-team-moss-tts-local-transformer-v1-5"    = 139
  "openmoss-team-moss-voicegenerator"                = 139
  "paddlepaddle-pp-docblocklayout"                   = 25
}

$passCount = 0; $failCount = 0; $failList = @()
function Write-Result([string]$text) { $text | Out-File -FilePath $RESULT -Append -Encoding utf8 }
$HEALTH_PY = "import urllib.request,sys;`ntry:`n r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3);`n sys.exit(0 if b'ok' in r.read() else 1)`nexcept Exception:`n sys.exit(1)"

Write-Result "========================================"
Write-Result "Retest remaining 6 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (offline, --network none)"
Write-Result "========================================"

foreach ($model in $MODELS.Keys) {
    $startup = $MODELS[$model]
    $maxWait = $startup * 2 + 60
    $mount   = "$ROOT\models\$model"

    Write-Host ">>> [$model] (max wait ${maxWait}s)"
    docker rm -f $model 2>$null | Out-Null
    $runOut = docker run -d --name $model --gpus all --network none `
        -v "${mount}:/workspace/models/${model}" `
        -w "/workspace/models/${model}" `
        $IMAGE bash start.sh 2>&1
    if ($LASTEXITCODE -ne 0) {
        $failCount++; $failList += $model
        Write-Result "[FAIL] $model | docker run failed: $runOut"
        continue
    }

    $ready = $false; $elapsed = 0
    while ($elapsed -lt $maxWait) {
        Start-Sleep -Seconds 5; $elapsed += 5
        docker exec $model python3 -c $HEALTH_PY 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        if ((docker inspect -f '{{.State.Running}}' $model 2>$null) -ne "true") { break }
    }

    if ($ready) {
        $testOut = docker exec $model python3 test/test.py 2>&1 | ForEach-Object { $_.ToString() }
        $testExit = $LASTEXITCODE
        if ($testExit -eq 0) {
            $passCount++
            Write-Host "  [PASS] $model (${elapsed}s)"
            Write-Result "[PASS] $model | startup ${elapsed}s | test PASS"
        } else {
            $failCount++; $failList += $model
            Write-Host "  [FAIL] $model (exit=$testExit)"
            Write-Result "[FAIL] $model | startup ${elapsed}s | test FAIL (exit=$testExit)"
        }
        foreach ($line in $testOut) { Write-Result "    $line" }
    } else {
        $failCount++; $failList += $model
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
Write-Result "Retest6 Summary: PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) { Write-Result "Retest6 Failed: $($failList -join ', ')" }
Write-Result "========================================"
Write-Host "Done. PASS $passCount / FAIL $failCount"

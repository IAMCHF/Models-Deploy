# ============================================================
# run_all_host_test.ps1 - Host-side sequential test for all models
# One model per container: start -> wait health -> run test.py -> stop/rm
# Results appended to result-0815.txt
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_all_host_test.ps1
# ============================================================

$ErrorActionPreference = "Continue"

$ROOT   = "d:\ssd-projects\Models-Deploy"
$RESULT = "$ROOT\result-0815.txt"
$IMAGE  = "models-deploy-base:latest"
$PORT   = 8080
$BASE   = "http://localhost:8080"

# model name -> startup time (seconds) from previous tests
$MODELS = [ordered]@{
  "alibaba-nlp-gte-modernbert-base"                          = 15
  "aratako-miocodec-25hz-44-1khz-v2"                         = 39
  "autogluon-chronos-2"                                      = 39
  "autogluon-mitra-classifier"                               = 9
  "bytedance-research-timer-s1"                              = 21
  "dleemiller-finecat-nli-l"                                 = 21
  "docling-project-codeformulav2"                            = 19
  "facebook-vjepa2-vitl-fpc64-256"                           = 39
  "google-timesfm-2-5-200m-transformers"                     = 19
  "google-videoprism-lvt-base-f16r288"                       = 35
  "ibm-granite-granite-speech-4-1-2b"                        = 17
  "ibm-granite-granite-timeseries-patchtst-fm-r1"            = 9
  "ibm-research-ttm-r3"                                      = 9
  "jhu-clsp-mmbert-base"                                     = 19
  "jusperlee-tiger-dnr"                                      = 23
  "k-iwa-time-anchor-modernbert-32m"                         = 7
  "koreapeter-ms-eff-gcvit-deepfake-b0-ff-plus-plus"         = 101
  "koreapeter-ms-eff-gcvit-deepfake-b5-ff-plus-plus"         = 101
  "microsoft-vibevoice-asr-hf"                               = 101
  "mldi-lab-kairos-23m"                                      = 25
  "mongodb-mdbr-leaf-ir"                                     = 7
  "neoquasar-kronos-base"                                    = 7
  "numind-nuextract3-fp8"                                    = 11
  "opengvlab-videomaev2-base"                                = 29
  "openmoss-team-moss-tts-local-transformer-v1-5"            = 139
  "openmoss-team-moss-voicegenerator"                        = 139
  "paddlepaddle-pp-chart2table"                              = 25
  "paddlepaddle-pp-docblocklayout"                           = 23
  "paddlepaddle-pp-doclayout-plus-l"                         = 25
  "paddlepaddle-pp-ocrv6-medium-det-onnx"                    = 1
  "paddlepaddle-pp-ocrv6-small-det-onnx"                     = 1
  "paddlepaddle-pp-ocrv6-small-rec-onnx"                     = 1
  "prior-labs-tabpfn-v2-clf"                                 = 11
  "prior-labs-tabpfn-v2-reg"                                 = 11
  "skywork-skywork-reward-v2-qwen3-0-6b"                     = 27
  "synthefy-nori-30m"                                        = 5
  "voyageai-voyage-4-nano"                                   = 35
  "weborganizer-topicclassifier-nourl"                       = 9
  "yuchenshen-fomo-0d"                                       = 13
}

$passCount = 0
$failCount = 0
$failList  = @()

function Write-Result([string]$text) {
    $text | Out-File -FilePath $RESULT -Append -Encoding utf8
}

# ---- header ----
Write-Result "========================================"
Write-Result "Model test - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (offline)"
Write-Result "Total: $($MODELS.Count) models"
Write-Result "========================================"

foreach ($model in $MODELS.Keys) {
    $startup = $MODELS[$model]
    $maxWait = $startup * 2 + 60
    $mount   = "$ROOT\models\$model"

    Write-Host ""
    Write-Host ">>> [$model] start container (max wait ${maxWait}s)"

    # ---- free port 8080: stop/remove any container publishing it ----
    $conflict = docker ps -q --filter "publish=$PORT" 2>$null
    if ($conflict) {
        foreach ($cid in $conflict) {
            docker stop $cid 2>$null | Out-Null
            docker rm  $cid 2>$null | Out-Null
        }
    }

    # ---- remove leftover container with same name ----
    docker rm -f $model 2>$null | Out-Null

    # ---- start container ----
    $runOut = docker run -d --name $model --gpus all -p "${PORT}:${PORT}" `
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

    # ---- wait for /health ----
    $ready  = $false
    $elapsed = 0
    while ($elapsed -lt $maxWait) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $health = curl.exe -s -m 3 "$BASE/health" 2>$null
        if ($health -match '"ok"') {
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
        Write-Host ">>> [$model] run test.py"
        $testOut = & python "$ROOT\models\$model\test\test.py" 2>&1 | ForEach-Object { $_.ToString() }
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
        $log = docker logs $model 2>&1 | Select-Object -Last 15
        foreach ($line in $log) { Write-Result "    $line" }
    }

    # ---- stop and remove container ----
    docker stop $model 2>$null | Out-Null
    docker rm  $model 2>$null | Out-Null
    Start-Sleep -Seconds 3
}

# ---- summary ----
Write-Result "========================================"
Write-Result "Summary: PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) {
    Write-Result "Failed: $($failList -join ', ')"
}
Write-Result "========================================"
Write-Host ""
Write-Host "========================================"
Write-Host "Done. PASS $passCount / FAIL $failCount / TOTAL $($MODELS.Count)"
if ($failList.Count -gt 0) {
    Write-Host "Failed: $($failList -join ', ')"
}
Write-Host "Result file: $RESULT"
Write-Host "========================================"

$ErrorActionPreference = "Continue"
$models = @("paddlepaddle-pp-ocrv6-small-rec-onnx", "paddlepaddle-pp-ocrv6-small-det-onnx", "paddlepaddle-pp-ocrv6-medium-det-onnx")
$HEALTH_PY = "import urllib.request,sys;`ntry:`n r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3);`n sys.exit(0 if b'ok' in r.read() else 1)`nexcept Exception:`n sys.exit(1)"
foreach ($m in $models) {
    Write-Output "=========================================="
    Write-Output ">>> [$m]"
    docker rm -f "rt-$m" 2>$null | Out-Null
    docker run -d --name "rt-$m" --gpus all -p 8080:8080 `
        -v "d:/ssd-projects/Models-Deploy/models/${m}:/workspace/models/${m}" `
        -w "/workspace/models/${m}" `
        models-deploy-base:latest bash start.sh | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 5
        docker exec "rt-$m" python3 -c $HEALTH_PY 2>$null
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        $state = docker inspect -f '{{.State.Running}}' "rt-$m" 2>$null
        if ($state -ne "true") { break }
    }
    if (-not $ok) {
        Write-Output "[$m] FAIL: 健康检查失败, 日志:"
        docker logs --tail 6 "rt-$m" 2>&1 | ForEach-Object { Write-Output "  $_" }
        docker rm -f "rt-$m" | Out-Null
        continue
    }
    $out = docker exec "rt-$m" python3 test/test.py 2>&1
    $code = $LASTEXITCODE
    $out | Select-Object -Last 5 | ForEach-Object { Write-Output "  $_" }
    if ($code -eq 0) { Write-Output "[$m] PASS" } else { Write-Output "[$m] FAIL (exit=$code)" }
    docker rm -f "rt-$m" | Out-Null
}

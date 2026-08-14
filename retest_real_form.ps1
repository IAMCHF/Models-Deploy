# retest_real_form.ps1 - 按用户真实部署形态重测受影响模型
# 形态: docker run -d --gpus all -p 8080:8080 -v 单模型目录 -w 模型目录 base镜像 bash start.sh
$ErrorActionPreference = "Continue"
$models = @(
    "yuchenshen-fomo-0d",
    "paddlepaddle-pp-ocrv6-small-det-onnx",
    "paddlepaddle-pp-ocrv6-small-rec-onnx",
    "paddlepaddle-pp-ocrv6-medium-det-onnx",
    "numind-nuextract3-fp8",
    "opengvlab-videomaev2-base"
)
$results = @()
foreach ($m in $models) {
    Write-Output "=========================================="
    Write-Output ">>> [$m] 启动容器 (真实形态)"
    docker rm -f "rt-$m" 2>$null | Out-Null
    docker run -d --name "rt-$m" --gpus all -p 8080:8080 `
        -v "d:/ssd-projects/Models-Deploy/models/${m}:/workspace/models/${m}" `
        -w "/workspace/models/${m}" `
        models-deploy-base:latest bash start.sh | Out-Null

    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 5
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch {}
    }
    if (-not $ok) {
        Write-Output "[$m] FAIL: 健康检查超时(450s), 最近日志:"
        docker logs --tail 8 "rt-$m" 2>&1 | ForEach-Object { Write-Output "  $_" }
        $results += "FAIL $m (health timeout)"
        docker rm -f "rt-$m" | Out-Null
        continue
    }
    Write-Output "[$m] 服务健康, 执行 test.py ..."
    $out = docker exec "rt-$m" python3 test/test.py 2>&1
    $code = $LASTEXITCODE
    $out | Select-Object -Last 6 | ForEach-Object { Write-Output "  $_" }
    if ($code -eq 0) {
        Write-Output "[$m] PASS"
        $results += "PASS $m"
    } else {
        Write-Output "[$m] FAIL (exit=$code)"
        $results += "FAIL $m (test exit $code)"
    }
    docker rm -f "rt-$m" | Out-Null
}
Write-Output "=========================================="
Write-Output "汇总:"
$results | ForEach-Object { Write-Output "  $_" }

# 诊断脚本: 起4个D类FAIL模型容器, 抓服务端完整日志
$ErrorActionPreference = "Continue"
$ROOT = "d:\ssd-projects\Models-Deploy"
$OUT  = "$ROOT\diagnose-0815.txt"
$IMAGE = "models-deploy-base:latest"
$MODELS = @(
  "dleemiller-finecat-nli-l",
  "docling-project-codeformulav2",
  "autogluon-mitra-classifier",
  "paddlepaddle-pp-docblocklayout"
)
$HEALTH_PY = "import urllib.request,sys;`ntry:`n r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3);`n sys.exit(0 if b'ok' in r.read() else 1)`nexcept Exception:`n sys.exit(1)"

"===== Diagnose $(Get-Date -Format 'HH:mm:ss') =====" | Out-File $OUT -Append -Encoding utf8

foreach ($model in $MODELS) {
  $mount = "$ROOT\models\$model"
  "`n########## $model ##########" | Out-File $OUT -Append -Encoding utf8
  docker rm -f $model 2>$null | Out-Null
  docker run -d --name $model --gpus all --network none `
    -v "${mount}:/workspace/models/${model}" `
    -w "/workspace/models/${model}" `
    $IMAGE bash start.sh 2>&1 | Out-Null

  $ready = $false; $elapsed = 0
  while ($elapsed -lt 300) {
    Start-Sleep -Seconds 5; $elapsed += 5
    docker exec $model python3 -c $HEALTH_PY 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    if ((docker inspect -f '{{.State.Running}}' $model 2>$null) -ne "true") { break }
  }
  "startup: ready=$ready elapsed=${elapsed}s" | Out-File $OUT -Append -Encoding utf8

  if ($ready) {
    "--- test.py output ---" | Out-File $OUT -Append -Encoding utf8
    docker exec $model python3 test/test.py 2>&1 | Select-Object -Last 15 | Out-File $OUT -Append -Encoding utf8
  }
  "--- server logs (last 80) ---" | Out-File $OUT -Append -Encoding utf8
  docker logs $model 2>&1 | Select-Object -Last 80 | Out-File $OUT -Append -Encoding utf8

  docker stop $model 2>$null | Out-Null
  docker rm  $model 2>$null | Out-Null
}
"===== DONE =====" | Out-File $OUT -Append -Encoding utf8

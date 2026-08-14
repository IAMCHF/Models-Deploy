#!/bin/bash
# 生成有意义的测试媒体数据（在 models-deploy-base 容器内运行，用完整版 ffmpeg）
set -e
M=/workspace/models
SRC="$M/ibm-granite-granite-speech-4-1-2b/weights/multilingual_sample.wav"
FF="ffmpeg -hide_banner -v error -y"

# 1) granite-speech ASR: 12s 真实多语种语音 16kHz
$FF -i "$SRC" -t 12 -ar 16000 -ac 1 -c:a pcm_s16le "$M/ibm-granite-granite-speech-4-1-2b/test/sample.wav"

# 2) vibevoice ASR: 10s 16kHz
$FF -i "$SRC" -t 10 -ar 16000 -ac 1 -c:a pcm_s16le "$M/microsoft-vibevoice-asr-hf/test/sample.wav"

# 3) miocodec 编解码: 6s 44.1kHz
$FF -i "$SRC" -t 6 -ar 44100 -ac 1 -c:a pcm_s16le "$M/aratako-miocodec-25hz-44-1khz-v2/test/sample.wav"

# 4) tiger-dnr 音源分离: 人声(6s, 44.1kHz) + 三和弦拨弦式音乐底噪混合
$FF -i "$SRC" -t 6 -ar 44100 -ac 1 /tmp/speech44.wav
$FF \
  -f lavfi -i "aevalsrc=0.10*sin(2*PI*261.63*t)*exp(-4*mod(t\,0.5)):s=44100:d=6" \
  -f lavfi -i "aevalsrc=0.10*sin(2*PI*329.63*t)*exp(-4*mod(t+0.15\,0.5)):s=44100:d=6" \
  -f lavfi -i "aevalsrc=0.10*sin(2*PI*392.00*t)*exp(-4*mod(t+0.30\,0.5)):s=44100:d=6" \
  -i /tmp/speech44.wav \
  -filter_complex "[0:a][1:a][2:a]amix=inputs=3:normalize=0[m];[3:a][m]amix=inputs=2:duration=first:normalize=0[a]" \
  -map "[a]" -ac 1 -ar 44100 -c:a pcm_s16le "$M/jusperlee-tiger-dnr/test/sample.wav"

# 5) 视频模型: 财报文档图慢速推近 -> 语义视频 (16fps, 5s, 80帧)
IMG="$M/paddlepaddle-pp-docblocklayout/test/sample.png"
$FF -loop 1 -i "$IMG" -vf "zoompan=z='min(1+0.006*on\,1.45)':d=80:x='iw/2-(iw/zoom/2)':y='ih/3-(ih/zoom/3)':s=256x256:fps=16" -frames:v 80 -pix_fmt yuv420p -c:v mpeg4 -q:v 5 "$M/facebook-vjepa2-vitl-fpc64-256/test/sample.mp4"
$FF -loop 1 -i "$IMG" -vf "zoompan=z='min(1+0.006*on\,1.45)':d=80:x='iw/2-(iw/zoom/2)':y='ih/3-(ih/zoom/3)':s=288x288:fps=16" -frames:v 80 -pix_fmt yuv420p -c:v mpeg4 -q:v 5 "$M/google-videoprism-lvt-base-f16r288/test/sample.mp4"
$FF -loop 1 -i "$IMG" -vf "zoompan=z='min(1+0.006*on\,1.45)':d=80:x='iw/2-(iw/zoom/2)':y='ih/3-(ih/zoom/3)':s=224x224:fps=16" -frames:v 80 -pix_fmt yuv420p -c:v mpeg4 -q:v 5 "$M/opengvlab-videomaev2-base/test/sample.mp4"

echo "=== 结果 ==="
ls -la "$M/ibm-granite-granite-speech-4-1-2b/test/sample.wav" \
      "$M/microsoft-vibevoice-asr-hf/test/sample.wav" \
      "$M/aratako-miocodec-25hz-44-1khz-v2/test/sample.wav" \
      "$M/jusperlee-tiger-dnr/test/sample.wav" \
      "$M/facebook-vjepa2-vitl-fpc64-256/test/sample.mp4" \
      "$M/google-videoprism-lvt-base-f16r288/test/sample.mp4" \
      "$M/opengvlab-videomaev2-base/test/sample.mp4" | awk '{print $5, $NF}'

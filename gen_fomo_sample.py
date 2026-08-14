import json
import numpy as np

rng = np.random.default_rng(42)
seq, batch, feat = 5000, 1, 100

# 正常模式: 每个特征 = 相位错开的慢正弦 + 轻噪声
t = np.arange(seq, dtype=np.float64)
train = np.stack(
    [np.sin(t / 50.0 + f * 0.1) + 0.1 * rng.standard_normal(seq) for f in range(feat)],
    axis=-1,
)  # (5000, 100)

# 测试段: 200 步, 延续训练分布, 在 50-70 步给每 10 个特征注入异常尖峰
tq = np.arange(200, dtype=np.float64)
test = np.stack(
    [np.sin((t[-1] + 1 + tq) / 50.0 + f * 0.1) + 0.1 * rng.standard_normal(200) for f in range(feat)],
    axis=-1,
)  # (200, 100)
test[50:70, ::10] += 6.0

payload = {
    "train_x": train[:, None, :].tolist(),  # (5000, 1, 100)
    "test_x": test[:, None, :].tolist(),    # (200, 1, 100)
}
out = "/workspace/models/yuchenshen-fomo-0d/test/sample.json"
with open(out, "w") as f:
    json.dump(payload, f)
print("已生成:", out, "train:", train.shape, "test:", test.shape)

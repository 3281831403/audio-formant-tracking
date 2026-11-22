# audio-formant-tracking

## 题面

给变声器加共振峰跟踪后，F1 在帧间乱跳、合成回去还全是杂音：1）`autocorr` 只算到 `order-1`，最后一阶缺失，`levinson` 少一阶；2）`levinson` 的反射系数符号反了，预测系数整体取反，合成信号反相；3）`lpc_roots` 把 `|z| > 1` 的根也保留，不稳定极点频率看似合理但带宽为负；4）`poles_to_formants` 的频率把 `arg` 的弧度直接当结果，漏乘 `sample_rate/(2π)`；5）`poles_to_formants` 的带宽用 `log(|z|)` 不取负，带宽为负；6）`track_formants` 每帧独立排序，共振峰标签在帧间跳变，连续性断言失败；7）`track_formants` 匹配时不做频率门限，跨度 1 kHz 也强行匹配；8）`synthesize` 用极点直接 IIR，但初始状态未预热，前 20 个样本异常，残差断言失败；9）`validate_poles` 不检查共轭成对，实数极点被当作共振峰；10）`track_batch` 复用 LPC 状态，第二条轨道从第一条的最后一帧继续匹配。难点是极点：稳定性、带宽符号、标签连续性三处错一个，跟踪就散。formant/lpc.py 是核心，tests/test_lpc.py 15个红。要修的是：极点稳定性、共轭配对、频率/带宽换算、标签连续性与批量状态隔离。

## 运行方式

从仓库根目录运行：

```bash
python -m pytest tests -q
```

初始状态必须恰好是：

```text
15 failed
```

不得出现 collection error、import error 或 syntax error。环境离线，测试不使用随机数，也不访问网络。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只能使用 Python 标准库（`math`、`cmath`）和 pytest，离线运行，无随机。
- `formant/lpc.py` 是核心实现文件，只写实现；`tests/test_lpc.py` 的 15 个测试不可修改。
- 8 个公开函数当前全部抛出 `NotImplementedError`；每个 docstring 写明阶数、频率 Hz、带宽 Hz 和 1e-6 容差等契约。
- 最终需让 15 个测试全部通过，尤其保证极点稳定、带宽为正、标签连续、合成预热正确、批量轨道互不串状态。

# LingBot-VLA V2 × RoboTwin 2.0

蚂蚁灵波比赛项目记录，覆盖 Clean50 数据准备、全量微调、RoboTwin 闭环评测及成果展示。

| 参赛评测条件 | 成功回合 | 成功率 |
| --- | ---: | ---: |
| Clean / seen | 4156 / 5000 | **83.12%** |
| Random / unseen | 2414 / 5000 | **48.28%** |

参赛资料记录的评测模型为 **20k + 2k**。每种环境均为 50 项任务 × 100 回合，FP32 推理，动作 chunk 50。Random 结果含 1,000 个已验证回合的复用。总数由逐任务表复核，原始逐回合日志与对应 checkpoint 未随本仓库发布。

- [演示视频发布页](https://github.com/peterpark494/lingbot/releases)
- [参赛记录与对照实验](docs/competition.md)
- [50 项任务明细](results/tasks.csv)
- [训练 Notebook](training/colab_full10.ipynb) · [单卡训练说明](docs/training.md)
- [RoboTwin 评测步骤](docs/evaluation.md)

## 项目流程

1. 从 [RoboTwin 2.0 数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) 获取 Aloha-AgileX Clean50 示范。读取多视角图像、关节状态、动作目标和官方指令。
2. 参赛记录的训练分两段：基础增强、均匀任务采样和低运动窗口降权训练至 20k，再加丰富图像增强并以较小学习率继续训练。正式评测选用续训 2k 的版本。
3. 本仓库同时提供另一次十任务单卡全量微调工程，包含数据转换、在线增强、训练和断点保存，具体配置和实测进度见训练说明。
4. 用 RoboTwin 执行闭环仿真。每个回合按任务完成条件判定成功，分别统计标准场景与随机化场景。

模型由视觉编码器、视觉语言主干和动作专家组成。单卡工程训练参与动作预测的权重。官方基础权重和官方 RoboTwin 后训练权重均可按评测说明取得；仓库不附参赛 checkpoint，也未用其他权重替代它。

## 校验结果表

```bash
python scripts/summarize_results.py
```

代码、模型与数据上游见 [LingBot-VLA V2](https://github.com/Robbyant/lingbot-vla-v2) 和 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)。使用上游资源时遵循相应许可。


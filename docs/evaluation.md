# RoboTwin 闭环评测

按 [LingBot 官方 RoboTwin 安装说明](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/experiment/robotwin/README.md) 配置 Linux、CUDA、RoboTwin 资产与环境。模型有两个官方入口：

- [LingBot-VLA V2 6B 基础权重](https://huggingface.co/robbyant/lingbot-vla-v2-6b)，用于初始化训练。
- [LingBot-VLA V2 6B RoboTwin 官方后训练权重](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)，用于运行官方流程。该模型由上游发布，具有自己的评测结果。

从官方后训练仓库获取导出的 `checkpoints/global_step_50000/hf_ckpt/`，另准备 Qwen3-VL-4B-Instruct 所需处理器文件。执行下例前将变量替换成实际路径。在 LingBot 官方源码目录分别用两个配置运行：

```bash
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export EVAL_WORKDIR=/path/to/RoboTwin
export MODEL=/path/to/checkpoints/global_step_50000/hf_ckpt

for mode in demo_clean demo_randomized; do
  bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$MODEL" \
    --output_base "$PWD/eval-output/$mode" \
    --task_config "$mode" \
    --num_gpus 1 --num_per_gpu 1 || exit 1
done
```

使用同一推理精度、动作 chunk、评测回合数和种子才能比较 checkpoint。正式记录包含 Clean/seen、Random/unseen；发布新结果时应记录环境配置、指令类别、权重版本和每项任务的成功回合数。仓库中的汇总脚本用于核对参赛表，不调用仿真器。

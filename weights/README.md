# LingBot-VLA V2 RoboTwin 权重

这里提供一套完整的 **Robbyant 官方 RoboTwin 后训练权重**的下载入口。模型为 6B，checkpoint 为 `global_step_50000`，共 6 个 Safetensors 分片，连同配置和处理器文件约 23.8 GiB。

[查看固定版本的官方文件](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin/tree/0451855729ec904f970600e0aec8b84661423afe/checkpoints/global_step_50000/hf_ckpt) · [仓库评测指南](../docs/evaluation.md)

在本仓库根目录执行：

```bash
python -m pip install huggingface_hub
python scripts/download_robotwin_weights.py
```

默认输出到 `weights/robbyant-robotwin/checkpoints/global_step_50000/hf_ckpt`。Windows 默认保存到 `D:\H bot\exports\robbyant-robotwin` 下，可用 `HBOT_LOCAL_DATA_ROOT` 或 `--output` 覆盖根目录。把脚本最终打印的 `hf_ckpt` 路径传给评测脚本。

使用 `--manifest-only` 可查看版本、下载体积和目标路径。下载脚本按 [清单](robotwin.json) 核对所有文件大小及官方已提供的 SHA-256 值。大文件由 Hugging Face 托管，GitHub 保存版本与校验清单。README 中的 83.12% 和 48.28% 来自参赛记录中的 20k + 2k 模型。

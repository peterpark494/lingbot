"""固定版本 LingBot 策略主干：SDPA、可反传 MoE、联合层梯度检查点。

不注入 LoRA，不加载旧 adapter，不加载深度/视频教师。
FP32 可训练权重 + BF16 autocast；8-bit 优化器在训练入口创建。
"""
import functools
import ast
import shutil
import struct
import sys
import types
from pathlib import Path

from colab10_common import read_json


def defer_official_data_imports(path):
    """只需 data.constants 时不提前导入 LeRobot；保留原始公开导出的按需访问。"""
    original = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(original)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_COLAB_DEFERRED_EXPORTS' for t in node.targets):
            assert 'build_vla_dataset' in ast.literal_eval(node.value)
            return
    exports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            for alias in node.names:
                assert alias.name != '*'
                exports[alias.asname or alias.name] = ('.' * node.level + node.module, alias.name)
    assert 'build_vla_dataset' in exports and 'build_dataloader' in exports
    header = original.split('from .chat_template import build_chat_template')[0]
    lazy = ("from importlib import import_module\n"
            f"_COLAB_DEFERRED_EXPORTS = {exports!r}\n"
            "__all__ = list(_COLAB_DEFERRED_EXPORTS)\n\n"
            "def __getattr__(name):\n"
            "    if name not in _COLAB_DEFERRED_EXPORTS:\n"
            "        raise AttributeError(name)\n"
            "    module, attribute = _COLAB_DEFERRED_EXPORTS[name]\n"
            "    value = getattr(import_module(module, __name__), attribute)\n"
            "    globals()[name] = value\n"
            "    return value\n")
    Path(path).write_text(header + '# Colab NPY loader: defer unused dataset dependencies.\n' + lazy, encoding='utf-8')


def sdpa_joint_attention(query, key, value, attention_mask, scaling=None):
    import torch
    import torch.nn.functional as F
    dtype = torch.bfloat16 if query.is_cuda else query.dtype
    q, k, v = [x.transpose(1, 2).to(dtype) for x in (query, key, value)]
    assert q.shape[1] % k.shape[1] == 0
    if q.shape[1] != k.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    mask = attention_mask[:, None] if attention_mask is not None and attention_mask.ndim == 3 else attention_mask
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0., scale=scaling)
    return out.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], -1).to(query.dtype)


def torch_moe(self, module, num_experts, routing_weights, selected_experts, hidden_states):
    import torch
    import torch.nn.functional as F
    output = torch.zeros_like(hidden_states)
    # unbind 让每个大参数只汇总一次各专家的梯度，保持官方 fused 权重布局。
    weights = zip(self.gate_proj.unbind(0), self.up_proj.unbind(0), self.down_proj.unbind(0))
    for expert, (gate_weight, up_weight, down_weight) in enumerate(weights):
        rows, slots = torch.where(selected_experts == expert)
        if rows.numel() == 0:
            continue
        x = hidden_states[rows]
        y = F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)
        output = output.index_add(0, rows, y.to(output.dtype) * routing_weights[rows, slots, None])
    return output


def checkpointed_joint_forward(self, attention_mask=None, position_ids=None, vlm_position_ids=None,
                               past_key_values=None, inputs_embeds=None, use_cache=None,
                               fill_kv_cache=None, ada_cond=None, visual_pos_masks=None,
                               deepstack_visual_embeds=None):
    """对官方联合 VLM/action 层整体重计算，包含层间联合注意力。"""
    import torch
    from torch.utils.checkpoint import checkpoint
    # 验证和后续推理沿用官方路径，包括 prefix-only / suffix-only KV cache。
    if not self.training or not torch.is_grad_enabled():
        return self._colab_original_forward(
            attention_mask=attention_mask, position_ids=position_ids, vlm_position_ids=vlm_position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            fill_kv_cache=fill_kv_cache, ada_cond=ada_cond, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds)
    assert not use_cache and past_key_values is None
    assert all(x is not None for x in inputs_embeds)
    models = [self.qwenvl.model.language_model, self.qwen_expert.model]
    hidden_vlm, hidden_action = inputs_embeds
    router_logits_list = []

    def layer(vlm, action, layer_idx):
        hidden = [vlm, action]
        qs, ks, vs = [], [], []
        for i, x in enumerate(hidden):
            kwargs = {'ada_cond': ada_cond} if i == 1 else {}
            q, k, v = models[i].layers[layer_idx](x, compute_kqv=True, **kwargs)
            qs.append(q.float())
            ks.append(k.float())
            vs.append(v.float())
        q, k = self.apply_mrope(torch.cat(qs, 1), torch.cat(ks, 1), position_ids)
        attended = self.attention_interface(q, k, torch.cat(vs, 1), attention_mask)
        boundary = vlm.shape[1]
        out_vlm = models[0].layers[layer_idx](vlm, attended, 0, boundary, output_atten=True)
        out_vlm = self._apply_deepstack(out_vlm, layer_idx, visual_pos_masks, deepstack_visual_embeds)
        out_action, logits = models[1].layers[layer_idx](
            action, attended, boundary, boundary + action.shape[1], output_atten=True, ada_cond=ada_cond)
        return out_vlm, out_action, logits

    assert len(models[0].layers) == len(models[1].layers)
    for index in range(len(models[0].layers)):
        # partial 固定层号，避免反向重计算时所有闭包都变成最后一层。
        hidden_vlm, hidden_action, logits = checkpoint(
            functools.partial(layer, layer_idx=index), hidden_vlm, hidden_action,
            use_reentrant=False, preserve_rng_state=True)
        if logits is not None:
            router_logits_list.append(logits)
    out_vlm = models[0].norm(hidden_vlm)
    if self.config.final_norm_adanorm:
        out_action, _ = models[1].norm(hidden_action, ada_cond)
    else:
        out_action = models[1].norm(hidden_action)
    return [out_vlm, out_action], None, router_logits_list


def shape_index(folder):
    import json
    index = read_json(folder / 'model.safetensors.index.json')
    shapes = {}
    for shard in sorted(set(index['weight_map'].values())):
        with (folder / shard).open('rb') as file:
            length = struct.unpack('<Q', file.read(8))[0]
            assert 0 < length < 64 * 1024**2
            header = json.loads(file.read(length))
        shapes.update({k: v['shape'] for k, v in header.items() if k != '__metadata__'})
    return index, shapes


def build_model(cfg, root, weights=None, saved_config=None):
    import torch
    import yaml
    from accelerate import init_empty_weights
    from safetensors import safe_open
    root = Path(root)
    canonical = root / 'source/official'
    source = root / 'source/runtime'
    # 每次从固定的官方源码生成运行副本，绝不在已补丁的文本上重复替换。
    shutil.copytree(canonical, source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    defer_official_data_imports(source / 'lingbotvla/data/__init__.py')
    path = source / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
    text = path.read_text(encoding='utf-8')
    for key in ['vlm_config', 'vlm_config.text_config', 'self.config.qwen_expert_config']:
        old = f'{key}._attn_implementation = "flash_attention_2"'
        assert text.count(old) == 1, f'官方源码版本不匹配：{old}'
        text = text.replace(old, f'{key}._attn_implementation = "eager"')
    path.write_text(text, encoding='utf-8')
    sys.path.insert(0, str(source))
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch, Qwen2FusedExperts
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    Qwen2FusedExperts.forward = torch_moe
    folder = Path(weights) if weights else root / 'models/base'
    index, shapes = shape_index(folder)
    if saved_config:
        config_args = read_json(Path(saved_config) / 'config.json')
    else:
        template = yaml.safe_load((source / 'configs/vla/robotwin/robotwin.yaml').read_text())
        config_args = {**template['model'], **template['train']}
    config_args.update(
        tokenizer_path=str(root / 'models/processor'), vlm_repo_id=None,
        align_params={}, use_compile=False, use_cache=False, attention_implementation='eager',
        vit_attn_implementation='sdpa', moe_implementation='fused',
        freeze_vision_encoder=False, train_expert_only=False, action_fp32=False,
        chunk_size=cfg['horizon'], tokenizer_max_length=cfg['max_text_length'],
        # 保留官方 MoE 平衡和 z-loss；bias_update_speed=0 避免重计算修改路由。
        sequence_wise_loss_coeff=.001, router_z_loss_coeff=.0001, bias_update_speed=0,
        precompute_grid_thw=False, use_qwen3_fixed_grid_cache=False,
        action_dim=shapes['model.action_out_proj.weight'][0],
        max_action_dim=shapes['model.action_in_proj.weight'][1],
        max_state_dim=shapes['model.state_proj.weight'][1])
    config = LingbotVLAV2Config(**config_args)
    with init_empty_weights(include_buffers=False):
        model = LingbotVlaV2Policy(config)
    expected = model.state_dict()
    parameters = set(dict(model.named_parameters()))
    missing = parameters - set(shapes)
    mismatches = [k for k, v in expected.items() if k in shapes and list(v.shape) != shapes[k]]
    assert not missing and not mismatches, f'主干参数缺失/形状不匹配：{list(missing)[:5]}, {mismatches[:5]}'
    ignored = sorted(set(shapes) - set(expected))
    assert not any(k.startswith('model.qwenvl_with_expert.') for k in ignored), ignored[:5]
    assert all(any(x in k for x in ['depth', 'video', 'task', 'future', 'query', 'align']) for k in ignored), ignored[:5]
    loaded = set()
    for shard in sorted(set(index['weight_map'].values())):
        state = {}
        with safe_open(folder / shard, framework='pt', device='cpu') as file:
            for name in file.keys():
                if name in expected:
                    tensor = file.get_tensor(name)
                    # FP32 参数避免小学习率更新被 BF16 权重舍入吞掉。
                    state[name] = tensor.to(device='cuda', dtype=torch.float32) if tensor.is_floating_point() else tensor.cuda()
                    loaded.add(name)
        model.load_state_dict(state, strict=False, assign=True)
        del state
        print('已加载权重分片：', shard, flush=True)
    assert parameters <= loaded and not any(p.is_meta for p in model.parameters())
    del expected
    model.to('cuda')
    model.requires_grad_(True)
    joint = model.model.qwenvl_with_expert
    joint.attention_interface = sdpa_joint_attention
    joint._colab_original_forward = joint.forward
    joint.forward = types.MethodType(checkpointed_joint_forward, joint)
    # 只为 vision blocks 开启 HF checkpoint；联合文本/动作层由上面的整体包装处理。
    from torch.utils.checkpoint import checkpoint
    for block in joint.qwenvl.visual.blocks:
        block.gradient_checkpointing = True
        block._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)
    # 关闭训练重计算中不参与损失的路由计数副作用；可微平衡损失保留。
    import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as expert_module
    expert_module._update_moe_runtime_stats = lambda *args, **kwargs: None
    # lm_head 不参与动作损失；不对它虚报全量更新。
    unused = []
    for name, parameter in model.named_parameters():
        if '.qwenvl.lm_head.' in name or '.qwen_expert.lm_head.' in name:
            parameter.requires_grad_(False)
            unused.append(name)
    model.train()
    metadata = {'ignored_inactive_auxiliary_keys': ignored, 'inactive_lm_heads': unused,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16',
                'checkpointing': 'joint VLM/action layers + vision blocks',
                'moe': 'differentiable torch experts with original fused weight layout',
                'teachers': False, 'lora': False}
    return model, metadata


def optimizer_groups(model, cfg):
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if '.qwenvl.model.visual.' in name:
            role = 'vision'
        elif '.qwenvl.' in name:
            role = 'vlm'
        else:
            role = 'action'
        decay = parameter.ndim > 1 and not name.endswith('.bias') and 'norm' not in name.lower()
        group = grouped.setdefault((role, decay), {'params': [], 'names': [],
            'lr': cfg[f'lr_{role}'], 'weight_decay': cfg['weight_decay'] if decay else 0., 'role': role})
        group['params'].append(parameter)
        group['names'].append(name)
    return list(grouped.values())

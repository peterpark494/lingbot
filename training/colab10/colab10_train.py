"""十任务策略主干全量微调。直接进行正式更新，无单独的冒烟训练。"""
import argparse
import gc
import math
import os
import shutil
import time
import uuid
from pathlib import Path

from colab10_common import (BatchCollator, EpisodeStore, UpdateSampler, WindowDataset,
                            identity, read_json, remove_owned, seed_all, selections, write_json)
from colab10_checkpoint import (finalize, load_optimizer, publish, resolve_pointer,
                                save_optimizer, save_weights, verify)
from colab10_model import build_model, optimizer_groups


def to_gpu(batch):
    import torch
    return {k: v.to('cuda', dtype=torch.bfloat16 if k in ['images', 'state', 'actions'] else v.dtype,
                    non_blocking=True) for k, v in batch.items()}


def lr_factor(step, cfg):
    # scheduler.step() 在一次更新后调用；第一个更新使用 1/warmup 的学习率。
    if step < cfg['warmup_steps']:
        return (step + 1) / max(1, cfg['warmup_steps'])
    progress = min(1., (step - cfg['warmup_steps']) / max(1, cfg['steps'] - cfg['warmup_steps']))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def signature(cfg, manifest, stats):
    keys = ['tasks', 'data_revision', 'source_revision', 'base_revision', 'processor_revision',
            'seed', 'horizon', 'max_text_length', 'effective_batch', 'steps', 'warmup_steps',
            'lr_vision', 'lr_vlm', 'lr_action', 'weight_decay', 'validation_episodes']
    return identity({'config': {k: cfg[k] for k in keys}, 'data': manifest, 'normalization': stats,
                     'augmentation': 'color_noise_blur_v1', 'training_code': code_identity()})


def code_identity():
    from colab10_common import digest
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('colab10_*.py'))}


def make_loader(cfg, manifest, stats, first, micro):
    import torch
    options = dict(dataset=WindowDataset(cfg['local_root'], manifest, stats, cfg['horizon']),
                   batch_sampler=UpdateSampler(manifest, cfg, first, micro),
                   collate_fn=BatchCollator(Path(cfg['local_root']) / 'models/processor', cfg['max_text_length']),
                   num_workers=cfg['workers'], pin_memory=True)
    if cfg['workers']:
        options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(**options)


def backward_update(model, optimizer, iterator, chosen, manifest, cfg, micro):
    import torch
    optimizer.zero_grad(set_to_none=True)
    total_valid = sum(min(cfg['horizon'], manifest['tasks'][t]['episodes'][str(e)]['frames'] - f)
                      for t, e, f, _ in chosen) * 14
    action_log, total_log = torch.zeros((), device='cuda'), torch.zeros((), device='cuda')
    for _ in range(cfg['effective_batch'] // micro):
        cpu = next(iterator)
        valid = float(cpu['joint_mask'].sum())
        batch = to_gpu(cpu)
        # 关闭 autocast 的全模型权重缓存，减少 FP32 训练时的 BF16 常驻副本。
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            result = model(**batch)
            action_loss, auxiliary = result[1], result[0] - result[1]
            loss = action_loss * (valid / total_valid) + auxiliary * (micro / cfg['effective_batch'])
        if not torch.isfinite(loss):
            raise FloatingPointError('训练 loss 出现 NaN/Inf，未执行这次参数更新')
        loss.backward()
        action_log += action_loss.detach().float() * (valid / total_valid)
        total_log += loss.detach().float()
    return float(action_log), float(total_log)


def validate(model, cfg, manifest, store, collate):
    import numpy as np
    import torch
    model.eval()
    scores = {}
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            for task_index, task in enumerate(cfg['tasks']):
                rng = np.random.default_rng([cfg['seed'], 999, task_index])
                values = []
                for index in range(cfg['validation_windows_per_task']):
                    episode = int(rng.choice(manifest['tasks'][task]['validation_episodes']))
                    n = manifest['tasks'][task]['episodes'][str(episode)]['frames']
                    key = (task, episode, int(rng.integers(n)), int(rng.integers(2**31)))
                    batch = to_gpu(collate([store.sample(key, augmentation=False)]))
                    # 验证始终 micro=1，固定每个窗口的 flow 时间和噪声，便于跨步比较。
                    torch.manual_seed(cfg['seed'] + task_index * 1000 + index)
                    batch['noise'] = torch.randn_like(batch['actions'])
                    batch['time'] = torch.rand(1, device='cuda') * .998 + .001
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        result = model(**batch)
                    value = float(result[1])  # 动作损失，不把路由正则混进验证指标。
                    if not math.isfinite(value):
                        raise FloatingPointError('验证 loss 出现 NaN/Inf')
                    values.append(value)
                scores[task] = sum(values) / len(values)
                print(f'  验证 {task}: action_loss={scores[task]:.6f}', flush=True)
        return {'per_task_action_loss': scores, 'mean_action_loss': sum(scores.values()) / len(scores),
                'worst_action_loss': max(scores.values()),
                'note': 'offline flow-matching loss, NOT simulation success rate'}
    finally:
        model.train()


def save_metadata(folder, model, cfg, manifest, stats, metadata):
    from transformers import AutoProcessor
    folder = Path(folder)
    model.config.save_pretrained(folder / 'model_config')
    AutoProcessor.from_pretrained(Path(cfg['local_root']) / 'models/processor',
                                  local_files_only=True).save_pretrained(folder / 'processor')
    for name, value in [('run_config.json', cfg), ('data_manifest.json', manifest),
                        ('normalization.json', stats), ('model_metadata.json', metadata)]:
        write_json(folder / name, value)
    code = folder / 'code'
    code.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('colab10_*.py'):
        shutil.copy2(p, code / p.name)


def save_checkpoint(model, optimizer, scheduler, cfg, manifest, stats, metadata, step,
                    micro, best_metric, validation, is_best, fingerprint):
    import torch
    run = Path(cfg['save_root']) / cfg['run_name']
    generation = uuid.uuid4().hex[:8]
    pending = run / f'resume-step-{step:06d}-{generation}.incomplete'
    pending.mkdir(parents=True)
    print(f'保存完整续训状态 step {step}（FP32 权重及 8-bit 优化器，写入 Drive 需要时间）', flush=True)
    save_weights(model, pending / 'weights')
    save_optimizer(optimizer, pending / 'optimizer')
    torch.save({'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state()}, pending / 'runtime.pt')
    save_metadata(pending, model, cfg, manifest, stats, metadata)
    write_json(pending / 'training_state.json', {'step': step, 'micro_batch': micro,
        'best_metric': best_metric if math.isfinite(best_metric) else None,
        'signature': fingerprint, 'validation': validation})
    finalize(pending, {'step': step, 'kind': 'full_resume', 'signature': fingerprint})
    if is_best:
        pending_best = run / f'best-step-{step:06d}-{generation}.incomplete'
        pending_best.mkdir()
        # 最佳模型面向 BF16 推理；精确续训必须读取 latest 的 FP32 权重。
        save_weights(model, pending_best / 'weights', dtype=torch.bfloat16)
        save_metadata(pending_best, model, cfg, manifest, stats, metadata)
        write_json(pending_best / 'validation.json', validation)
        finalize(pending_best, {'step': step, 'kind': 'bf16_inference_model', 'signature': fingerprint})
        best = publish(run, pending_best, 'best', step)
        print('最佳推理模型：', best, flush=True)
    latest = publish(run, pending, 'latest', step)
    print('续训状态已完整保存：', latest, flush=True)
    return latest


def train(cfg):
    import fcntl
    import numpy as np
    import torch
    import bitsandbytes as bnb
    root = Path(cfg['local_root'])
    run = Path(cfg['save_root']) / cfg['run_name']
    run.mkdir(parents=True, exist_ok=True)
    # 本地进程锁防止重复点击训练时启动第二个进程。
    lock = (root / 'training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '需要支持 BF16 的 NVIDIA GPU'
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert memory >= 75, f'本配置面向 80GB 显卡，当前只有 {memory:.1f} GiB'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(cfg['seed'])
    manifest, stats = read_json(root / 'data_manifest.json'), read_json(root / 'normalization.json')
    fingerprint = signature(cfg, manifest, stats)
    config_path = run / 'experiment.json'
    if config_path.exists():
        assert read_json(config_path)['signature'] == fingerprint, '实验配置/数据/代码变化，请换 run_name，不能静默接旧训练'
    else:
        write_json(config_path, {'signature': fingerprint, 'config': cfg})
    resume = resolve_pointer(run, 'latest') if cfg['resume'] == 'auto' else None
    if cfg['resume'] not in ['auto', None]:
        resume = Path(cfg['resume'])
    if resume is None and resolve_pointer(run, 'latest') is not None:
        raise ValueError('当前实验已有 checkpoint；从头训练请换 run_name')
    first, best_metric, micro = 0, float('inf'), cfg['micro_batch']
    if resume is not None:
        print('核对并读取完整续训状态：', resume, flush=True)
        verify(resume)
        saved = read_json(resume / 'training_state.json')
        assert saved['signature'] == fingerprint
        first = saved['step']
        best_folder = resolve_pointer(run, 'best')
        if best_folder is not None:
            best_metric = read_json(best_folder / 'validation.json')['mean_action_loss']
        micro = min(micro, saved['micro_batch'])
        if first >= cfg['steps']:
            print(f'这个实验已经完成 {first} 步，不会重新训练。新实验请换 run_name。', flush=True)
            return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    model, metadata = build_model(cfg, root, weights=resume / 'weights' if resume else None,
                                 saved_config=resume / 'model_config' if resume else None)
    groups = optimizer_groups(model, cfg)
    optimizer = bnb.optim.AdamW8bit(groups, betas=(.9, .95), eps=1e-8, min_8bit_size=4096,
                                   block_wise=True, percentile_clipping=100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    if resume:
        load_optimizer(optimizer, resume / 'optimizer')
        state = torch.load(resume / 'runtime.pt', map_location='cpu', weights_only=True)
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    print(f"GPU: {torch.cuda.get_device_name(0)} / {memory:.1f} GiB", flush=True)
    print(f"策略主干全量训练：{metadata['trainable_parameters']:,} 参数；无 LoRA，无辅助教师。", flush=True)
    print(f"正式训练 step {first + 1} → {cfg['steps']}，micro={micro}，"
          f"accum={cfg['effective_batch'] // micro}，有效 batch={cfg['effective_batch']}。", flush=True)
    print('FP32 参数/梯度、BF16 计算、8-bit AdamW，梯度检查点已接入联合层和视觉层。', flush=True)
    loader = make_loader(cfg, manifest, stats, first + 1, micro)
    iterator = iter(loader)
    valid_store = EpisodeStore(root, manifest, stats, cfg['horizon'])
    valid_collate = BatchCollator(root / 'models/processor', cfg['max_text_length'])
    local_log = root / 'train.jsonl'
    persistent_log = run / 'train.jsonl'
    # 日志可能已写到 600、完整 checkpoint 只有 500；续训前丢弃未提交分支，避免重复步号。
    for path in [persistent_log, run / 'validation.jsonl']:
        if path.exists():
            temporary = path.with_name(path.name + '.truncate.tmp')
            with path.open(encoding='utf-8') as inp, temporary.open('w', encoding='utf-8') as out:
                for line in inp:
                    try:
                        if __import__('json').loads(line)['step'] <= first:
                            out.write(line.rstrip('\n') + '\n')
                    except (ValueError, KeyError):
                        pass  # 意外中断留下的最后半行不是已提交记录。
            os.replace(temporary, path)
    if persistent_log.exists():
        shutil.copy2(persistent_log, local_log)
    else:
        local_log.write_text('', encoding='utf-8')
    pending_records, completed, last_saved = [], first, first
    unsafe_update = False
    checkpoint_failed = False
    last_validation = None
    update_times = []

    def flush_logs():
        nonlocal pending_records
        if pending_records:
            with persistent_log.open('a', encoding='utf-8') as file:
                file.write(''.join(pending_records))
                file.flush()
            pending_records = []

    def save_current(is_best=False):
        nonlocal last_saved, checkpoint_failed
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        flush_logs()
        try:
            save_checkpoint(model, optimizer, scheduler, cfg, manifest, stats, metadata, completed,
                            micro, best_metric, last_validation, is_best, fingerprint)
        except BaseException:
            checkpoint_failed = True
            raise
        last_saved = completed

    try:
        for step in range(first + 1, cfg['steps'] + 1):
            tick = time.monotonic()
            while True:
                retry = False
                try:
                    seed_all(cfg['seed'] + step)
                    chosen = selections(manifest, cfg, step)
                    action_loss, total_loss = backward_update(model, optimizer, iterator, chosen, manifest, cfg, micro)
                except torch.cuda.OutOfMemoryError as exc:
                    exc.__traceback__ = None
                    if micro == 1:
                        raise RuntimeError('micro=1 仍显存不足，当前完整全量配置不能在本运行时容纳；不会自动冻结参数') from None
                    retry = True
                if not retry:
                    break
                optimizer.zero_grad(set_to_none=True)
                del iterator, loader
                gc.collect()
                torch.cuda.empty_cache()
                micro //= 2
                loader = make_loader(cfg, manifest, stats, step, micro)
                iterator = iter(loader)
                print(f'反向显存不足：重放同一次更新，micro={micro}，'
                      f'accum={cfg["effective_batch"] // micro}，有效 batch 保持不变。', flush=True)
            if step == first + 1:
                for role in ['vision', 'vlm', 'action']:
                    has_gradient = any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                        for group in groups if group['role'] == role for p in group['params'])
                    assert has_gradient, f'{role} 没有有效梯度；停止，避免误把冻结训练当成全量训练'
                print('首个正式更新：视觉、VLM、动作分组均有梯度。', flush=True)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                 1., error_if_nonfinite=True)
            unsafe_update = True
            try:
                optimizer.step()
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError('优化器更新阶段显存不足，可能已有部分参数更新；不会保存当前状态。'
                                   '上次完整 latest 保持有效，需要重新规划优化器内存后续训。') from None
            scheduler.step()
            completed = step
            unsafe_update = False
            seconds = time.monotonic() - tick
            update_times.append(seconds)
            record = {'step': step, 'action_loss': action_loss, 'total_loss': total_loss,
                      'grad_norm': float(norm), 'seconds': seconds, 'micro_batch': micro,
                      'effective_batch': cfg['effective_batch'],
                      'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
                      'learning_rates': {g['role']: g['lr'] for g in optimizer.param_groups}}
            line = __import__('json').dumps(record) + '\n'
            with local_log.open('a', encoding='utf-8') as file:
                file.write(line)
            pending_records.append(line)
            if step == first + 1 or step % cfg['log_every'] == 0:
                flush_logs()
                average = sum(update_times[-20:]) / len(update_times[-20:])
                hours = average * (cfg['steps'] - step) / 3600
                print(f'step {step}/{cfg["steps"]} | action_loss={action_loss:.5f} | '
                      f'{seconds:.1f}s/update | peak={record["peak_gpu_gib"]:.1f}GiB | '
                      f'剩余纯更新约 {hours:.1f}h（不含验证/保存）', flush=True)
                write_json(run / 'status.json', {'status': 'training', **record})
            is_best = False
            if step % cfg['validate_every'] == 0 or step == cfg['steps']:
                optimizer.zero_grad(set_to_none=True)
                last_validation = {'step': step, **validate(model, cfg, manifest, valid_store, valid_collate)}
                with (run / 'validation.jsonl').open('a', encoding='utf-8') as file:
                    file.write(__import__('json').dumps(last_validation) + '\n')
                is_best = last_validation['mean_action_loss'] < best_metric
                if is_best:
                    best_metric = last_validation['mean_action_loss']
            if step % cfg['save_every'] == 0 or step == cfg['steps'] or is_best:
                save_current(is_best)
        write_json(run / 'status.json', {'status': 'completed', 'step': completed,
                   'best_offline_action_loss': best_metric,
                   'note': '训练完成；仿真成功率需后续在 RoboTwin 测量'})
    except BaseException as exc:
        # 只在已完成一次完整 optimizer.step、且下一步未部分写入参数时保存。
        if completed > last_saved and not unsafe_update and not checkpoint_failed:
            print('训练中断，保存最近完成的更新……', flush=True)
            save_current(False)
        flush_logs()
        write_json(run / 'status.json', {'status': 'interrupted_or_failed', 'step': completed,
                   'saved_step': last_saved, 'message': str(exc), 'partial_update_discarded': unsafe_update})
        raise
    finally:
        flush_logs()
        del iterator, loader
    print('正式训练已完成；最佳 BF16 推理模型与最近完整续训状态均已保存。', flush=True)
    print('结果目录：', run, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cfg = read_json(parser.parse_args().config)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    train(cfg)

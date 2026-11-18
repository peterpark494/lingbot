"""分片保存完整模型/优化器。只有写完 COMPLETE.json 才切换续训指针。"""
import os
from pathlib import Path

from colab10_common import digest, read_json, remove_owned, write_json


def tree_to(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    return value


def tree_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tree_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tree_bytes(v) for v in value)
    return 0


def save_weights(model, folder, dtype=None, shard_bytes=1024**3):
    from safetensors.torch import save_file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    part, size, index, total, number = {}, 0, {}, 0, 0

    def flush():
        nonlocal part, size, number
        if not part:
            return
        number += 1
        name = f'model-{number:05d}.safetensors'
        save_file(part, str(folder / name), metadata={'format': 'pt'})
        index.update({k: name for k in part})
        part, size = {}, 0

    for name, tensor in model.state_dict().items():
        target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
        bytes_ = tensor.numel() * __import__('torch').empty((), dtype=target_dtype).element_size()
        if part and size + bytes_ > shard_bytes:
            flush()
        part[name] = tensor.detach().to(device='cpu', dtype=target_dtype, copy=True).contiguous()
        size += bytes_
        total += bytes_
    flush()
    write_json(folder / 'model.safetensors.index.json',
               {'metadata': {'total_size': total}, 'weight_map': index})


def save_optimizer(optimizer, folder, shard_bytes=1024**3):
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state_dict = optimizer.state_dict()  # 引用 GPU 状态，不在这里复制整套。
    part, size, files = {}, 0, []

    def flush():
        nonlocal part, size
        if not part:
            return
        name = f'optimizer-{len(files) + 1:05d}.pt'
        torch.save(part, folder / name)
        files.append(name)
        part, size = {}, 0

    for key, state in state_dict['state'].items():
        bytes_ = tree_bytes(state)
        if part and size + bytes_ > shard_bytes:
            flush()
        part[key] = tree_to(state, 'cpu')
        size += bytes_
    flush()
    write_json(folder / 'optimizer_index.json',
               {'files': files, 'param_groups': state_dict['param_groups']})


def load_optimizer(optimizer, folder):
    import torch
    folder = Path(folder)
    index = read_json(folder / 'optimizer_index.json')
    assert len(index['param_groups']) == len(optimizer.param_groups)
    mapping = {}
    for current, saved in zip(optimizer.param_groups, index['param_groups']):
        assert current['names'] == saved['names'], '优化器参数顺序变化，不能续训'
        mapping.update(zip(saved['params'], current['params']))
        current.update({k: v for k, v in saved.items() if k != 'params'})
    optimizer.state.clear()
    for filename in index['files']:
        chunk = torch.load(folder / filename, map_location='cpu', weights_only=True)
        for key, state in chunk.items():
            parameter = mapping[key]
            optimizer.state[parameter] = tree_to(state, parameter.device)
        del chunk
    # 保持 bitsandbytes 原始 dtype（尤其 uint8 状态）；下一次 step 正常初始化管理器。


def finalize(folder, metadata):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in folder.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'}
    write_json(folder / 'COMPLETE.json', {**metadata, 'files': files})


def verify(folder):
    folder = Path(folder)
    complete = read_json(folder / 'COMPLETE.json')
    for name, record in complete['files'].items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder.resolve())
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], f'损坏的 checkpoint: {name}'
    return complete


def resolve_pointer(run, kind):
    run = Path(run).resolve()
    pointer = run / f'{kind}.json'
    if not pointer.is_file():
        return None
    folder = (run / read_json(pointer)['directory']).resolve()
    assert folder.is_relative_to(run) and folder != run
    return folder


def publish(run, folder, kind, step):
    run, folder = Path(run), Path(folder)
    assert (folder / 'COMPLETE.json').is_file()
    previous = resolve_pointer(run, kind)
    final = folder.with_name(folder.name.removesuffix('.incomplete'))
    assert not final.exists()
    os.replace(folder, final)
    write_json(run / f'{kind}.json', {'directory': final.name, 'step': step})
    if previous is not None and previous != final:
        remove_owned(previous, run)
    return final

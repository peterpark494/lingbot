"""下载固定版本的官方 clean 数据、转成本地训练缓存、计算训练集统计量。"""
import argparse
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from colab10_common import (CAMERAS, FORMAT, digest, identity, read_json,
                            remove_owned, split_episodes, write_json)


def download_models(cfg, root):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    records = {}
    for name, repo, revision in [
        ('base', 'robbyant/lingbot-vla-v2-6b', cfg['base_revision']),
        ('processor', 'Qwen/Qwen3-VL-4B-Instruct', cfg['processor_revision']),
    ]:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        entries = {entry.rfilename: entry for entry in info.siblings}
        folder = root / 'models' / name
        support = [p for p in entries if '/' not in p and p.endswith(
            ('.json', '.txt', '.model', '.jinja', '.tiktoken'))
            and p != 'model.safetensors.index.json']
        if name == 'base':
            index_file = hf_hub_download(repo, 'model.safetensors.index.json',
                                         revision=revision, local_dir=folder)
            shards = set(read_json(index_file)['weight_map'].values())
            wanted = sorted(set(support) | shards | {'model.safetensors.index.json'})
        else:
            wanted = sorted(support)  # 配置和分词器，不下载第二套 Qwen 模型。
        remaining = sum(entries[p].size for p in wanted if not (folder / p).is_file()
                        or (folder / p).stat().st_size != entries[p].size)
        if shutil.disk_usage(root).free < remaining + 8 * 1024**3:
            raise RuntimeError('本地磁盘不足，请使用 Colab 高 RAM 运行时并释放本项目无用文件')
        print(f'下载 {name}: {repo}@{info.sha}，尚需 {remaining / 1024**3:.2f} GiB', flush=True)
        snapshot_download(repo, revision=revision, allow_patterns=wanted,
                          local_dir=folder, max_workers=2)
        for p in wanted:
            assert (folder / p).stat().st_size == entries[p].size, p
        records[name] = {'repo': repo, 'revision': info.sha, 'files': wanted}
    return records


def download_source(cfg, root):
    source = root / 'source' / 'official'
    marker = source / '_download.json'
    if marker.is_file() and read_json(marker)['revision'] == cfg['source_revision']:
        return read_json(marker)
    archive = root / 'source.tar.gz'
    url = f"https://codeload.github.com/Robbyant/lingbot-vla-v2/tar.gz/{cfg['source_revision']}"
    print('下载固定版本 LingBot 官方源码', flush=True)
    urllib.request.urlretrieve(url, archive)
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as package:
        for member in package:
            relative = Path(*Path(member.name).parts[1:])
            target = (source / relative).resolve()
            if not target.is_relative_to(source.resolve()):
                raise ValueError('源码压缩包路径越界')
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as inp, target.open('wb') as out:
                    shutil.copyfileobj(inp, out)
    record = {'revision': cfg['source_revision'], 'archive_sha256': digest(archive)}
    write_json(marker, record)
    archive.unlink()
    return record


def read_states(file):
    import numpy as np
    left = np.asarray(file['joint_action/left_arm'], dtype=np.float32)
    right = np.asarray(file['joint_action/right_arm'], dtype=np.float32)
    n = len(left)
    assert n >= 2 and left.shape == right.shape == (n, 6)
    grips = [np.asarray(file[f'joint_action/{side}_gripper'], dtype=np.float32).reshape(n, 1)
             for side in ['left', 'right']]
    states = np.concatenate([left, grips[0], right, grips[1]], axis=1)
    assert states.shape == (n, 14) and np.isfinite(states).all()
    for camera in CAMERAS:
        assert len(file[f'observation/{camera}/rgb']) == n
    return states


def decode_rgb(value):
    import cv2
    import numpy as np
    from PIL import Image
    bits = value.tobytes() if hasattr(value, 'tobytes') else bytes(value)
    bgr = cv2.imdecode(np.frombuffer(bits, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('官方数据中的 RGB 解码失败')
    # 与已跑通的转换对齐：先统一为 640×480，再缩至模型的 224×224。
    bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))


def prepare_task(cfg, root, task):
    import h5py
    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'TianxingChen/RoboTwin2.0'
    filename = f'dataset/{task}/aloha-agilex_clean_50.zip'
    # 精确文件名白名单；不使用整个官方训练清单里的 randomized/Piper 数据。
    entries = list(HfApi().list_repo_tree(repo, path_in_repo=f'dataset/{task}',
                                         revision=cfg['data_revision'], repo_type='dataset'))
    selected = [entry for entry in entries if entry.path == filename]
    assert len(selected) == 1, f'没有指定的官方 Aloha clean50 文件：{filename}'
    entry = selected[0]
    if shutil.disk_usage(root).free < entry.size + 4 * 1024**3:
        raise RuntimeError(f'下载 {task} 前本地空间不足')
    archive = Path(hf_hub_download(repo, filename, repo_type='dataset',
                                 revision=cfg['data_revision'], local_dir=root / 'downloads'))
    assert archive.stat().st_size == entry.size
    archive_hash = digest(archive)
    lfs = getattr(entry, 'lfs', None)
    lfs_hash = getattr(lfs, 'sha256', None) if lfs is not None else None
    if isinstance(lfs, dict):
        lfs_hash = lfs.get('sha256')
    if lfs_hash:
        assert archive_hash == lfs_hash, 'ZIP 哈希不匹配'
    specification = {'format': FORMAT, 'archive_sha256': archive_hash,
                     'cameras': CAMERAS, 'size': [224, 224]}
    cache_id = identity(specification)
    destination = root / 'prepared' / task
    destination.mkdir(parents=True, exist_ok=True)
    episodes = {}
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        for episode in range(50):
            prefix = 'aloha-agilex_clean_50'
            hp, jp = f'{prefix}/data/episode{episode}.hdf5', f'{prefix}/instructions/episode{episode}.json'
            assert hp in names and jp in names, f'{task}/episode{episode} 不完整'
            instructions = __import__('json').loads(package.read(jp))
            seen = instructions.get('seen')
            assert isinstance(seen, list) and seen and all(isinstance(s, str) and s.strip() for s in seen)
            folder = destination / f'episode{episode}'
            marker = folder / 'complete.json'
            if marker.is_file():
                record = read_json(marker)
                if record['cache_id'] != cache_id:
                    raise ValueError(f'旧缓存与当前数据不同：{folder}；请修改 local_root')
                assert (folder / 'rgb.npy').stat().st_size == record['rgb_bytes']
                assert digest(folder / 'joints.npz') == record['joints_sha256']
                episodes[str(episode)] = record
                continue
            temporary = destination / f'.episode{episode}.incomplete'
            if temporary.exists():
                remove_owned(temporary, destination)
            temporary.mkdir()
            hdf5 = temporary / 'source.hdf5'
            with package.open(hp) as inp, hdf5.open('wb') as out:
                shutil.copyfileobj(inp, out)
            with h5py.File(hdf5, 'r') as file:
                states = read_states(file)
                n = len(states) - 1
                needed = n * 3 * 224 * 224 * 3
                if shutil.disk_usage(root).free < needed + 8 * 1024**3:
                    raise RuntimeError('224 RGB 缓存将写满本地盘，请换更大的 Colab 运行时磁盘')
                rgb = np.lib.format.open_memmap(temporary / 'rgb.npy', mode='w+',
                                               dtype=np.uint8, shape=(n, 3, 224, 224, 3))
                for t in range(n):
                    for i, camera in enumerate(CAMERAS):
                        rgb[t, i] = decode_rgb(file[f'observation/{camera}/rgb'][t])
                rgb.flush()
                del rgb
                np.savez(temporary / 'joints.npz', states=states[:-1], actions=states[1:])
            record = {'cache_id': cache_id, 'frames': n, 'seen': seen,
                      'rgb_bytes': (temporary / 'rgb.npy').stat().st_size,
                      'joints_sha256': digest(temporary / 'joints.npz')}
            hdf5.unlink()
            write_json(temporary / 'complete.json', record)
            assert not folder.exists()
            temporary.rename(folder)
            episodes[str(episode)] = record
            if episode % 10 == 0 or episode == 49:
                print(f'{task}: 已转换 {episode + 1}/50 条轨迹', flush=True)
    train, valid = split_episodes(task, cfg['validation_episodes'], cfg['seed'])
    return {'archive': filename, 'archive_sha256': archive_hash, 'episodes': episodes,
            'train_episodes': train, 'validation_episodes': valid,
            'train_frames': sum(episodes[str(i)]['frames'] for i in train),
            'validation_frames': sum(episodes[str(i)]['frames'] for i in valid)}


def prepare(cfg):
    import cv2
    import numpy as np
    root = Path(cfg['local_root'])
    root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    os.environ['HF_HOME'] = str(root / 'hf_cache')
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE']:
        os.environ.pop(key, None)
    manifest = {'format': FORMAT, 'data_revision': cfg['data_revision'], 'tasks': {}}
    for task in cfg['tasks']:
        manifest['tasks'][task] = prepare_task(cfg, root, task)
        write_json(root / 'data_manifest.json', manifest)
    states, actions = [], []
    for task, record in manifest['tasks'].items():
        for episode in record['train_episodes']:
            with np.load(root / 'prepared' / task / f'episode{episode}' / 'joints.npz', allow_pickle=False) as z:
                states.append(z['states'])
                actions.append(z['actions'])
    sl, sh = np.quantile(np.concatenate(states), [.01, .99], axis=0).astype(np.float32)
    al, ah = np.quantile(np.concatenate(actions), [.01, .99], axis=0).astype(np.float32)
    stats = {'s_low': sl.tolist(), 's_high': sh.tolist(), 'a_low': al.tolist(), 'a_high': ah.tolist(),
             'method': 'shared_train_only_q01_q99_without_clipping',
             'raw_to_model_slots': [0, 1, 2, 3, 4, 5, 28, 6, 7, 8, 9, 10, 11, 29]}
    write_json(root / 'normalization.json', stats)
    manifest['models'] = download_models(cfg, root)
    manifest['source'] = download_source(cfg, root)
    # 预先核对最长指令，不在正式训练中静默丢掉句尾。
    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(root / 'models/processor', local_files_only=True).tokenizer
    max_tokens = 0
    for record in manifest['tasks'].values():
        for ep in record['episodes'].values():
            for text in ep['seen']:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                      tokenize=False, add_generation_prompt=False)
                max_tokens = max(max_tokens, len(tokenizer.encode(prompt, add_special_tokens=False)))
    if max_tokens > cfg['max_text_length']:
        raise ValueError(f"最长指令 {max_tokens} tokens，请把 max_text_length 调到至少这个值后重跑本格")
    manifest['max_instruction_tokens'] = max_tokens
    manifest['train_frames'] = sum(t['train_frames'] for t in manifest['tasks'].values())
    write_json(root / 'data_manifest.json', manifest)
    # Drive 只保存轻量清单，不复制几十 GB 的 RGB 缓存；新会话按同一版本重新生成。
    persistent = Path(cfg['save_root']) / cfg['run_name']
    persistent.mkdir(parents=True, exist_ok=True)
    for filename in ['data_manifest.json', 'normalization.json']:
        target = persistent / filename
        if target.exists() and read_json(target) != read_json(root / filename):
            raise ValueError(f'{target} 已有不同实验，请换 run_name；不会覆盖')
        shutil.copy2(root / filename, target)
    print(f"\n准备完成：{len(cfg['tasks'])} 个任务，{manifest['train_frames']} 个训练帧。", flush=True)
    print(f"固定 {cfg['steps']} 次更新 × batch {cfg['effective_batch']} = "
          f"{cfg['steps'] * cfg['effective_batch']} 次窗口抽样。", flush=True)
    print('缓存格式：按 episode 的 RGB224 NPY + 14维关节 NPZ，训练时在线增广。', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    prepare(read_json(parser.parse_args().config))

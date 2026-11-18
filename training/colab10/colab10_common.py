"""Colab 十任务训练的公共函数。只在选定项目目录内写入。"""
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

CAMERAS = ['head_camera', 'left_camera', 'right_camera']
SLOTS = [0, 1, 2, 3, 4, 5, 28, 6, 7, 8, 9, 10, 11, 29]
FORMAT = 'robotwin_rgb224_next_joint_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def remove_owned(path, parent):
    """仅清理当前项目自己的临时文件/旧 checkpoint，不接收任意目录。"""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f'拒绝清理项目目录以外的路径：{path}')
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def pack(values, width=55):
    import numpy as np
    result = np.zeros((*values.shape[:-1], width), dtype=np.float32)
    result[..., SLOTS] = values
    return result


def split_episodes(task, count, seed):
    import numpy as np
    # 与任务列表顺序无关：换排列不改变每个任务的验证轨迹。
    salt = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng([seed, salt]).permutation(50)
    valid = sorted(order[:count].tolist())
    return sorted(order[count:].tolist()), valid


def selections(manifest, cfg, step):
    """每个 update 确定性抽样：任务→轨迹→起始帧，避免长轨迹独占。"""
    import numpy as np
    rng = np.random.default_rng([cfg['seed'], step])
    tasks = cfg['tasks']
    # 连续的抽样槽位轮流覆盖任务；10 个任务时连续 5 次更新各取 8 个样本。
    ids = (np.arange(cfg['effective_batch']) + (step - 1) * cfg['effective_batch']) % len(tasks)
    rng.shuffle(ids)
    result = []
    for position, task_id in enumerate(ids):
        task = tasks[int(task_id)]
        episodes = manifest['tasks'][task]['train_episodes']
        episode = int(rng.choice(episodes))
        n = manifest['tasks'][task]['episodes'][str(episode)]['frames']
        result.append((task, episode, int(rng.integers(n)),
                       int(rng.integers(0, 2**32 - 1))))
    return result


def augment_rgb(views, seed):
    """只增广未归一化的 RGB；三相机共享颜色参数，不改变几何和动作标签。"""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        return np.array(views, copy=True)
    brightness, contrast = rng.uniform(.85, 1.15, 2)
    saturation = rng.uniform(.9, 1.1)
    blur = rng.random() < .1
    sigma = rng.uniform(1., 3.) if rng.random() < .15 else 0.
    output = []
    for view in views:
        image = Image.fromarray(view)
        image = ImageEnhance.Brightness(image).enhance(float(brightness))
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
        image = ImageEnhance.Color(image).enhance(float(saturation))
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=.35))
        rgb = np.asarray(image).astype(np.float32)
        if sigma:
            rgb += rng.normal(0, sigma, rgb.shape)
        output.append(rgb.round().clip(0, 255).astype(np.uint8))
    return np.stack(output)


class EpisodeStore:
    """图像 mmap 按需读本地 SSD；不把 500 条轨迹预装进 RAM/GPU。"""
    def __init__(self, root, manifest, stats, horizon):
        self.root, self.manifest, self.stats, self.horizon = Path(root), manifest, stats, horizon
        self.arrays = {}

    def sample(self, key, augmentation=True):
        import numpy as np
        task, episode, t, seed = key
        name = (task, episode)
        if name not in self.arrays:
            folder = self.root / 'prepared' / task / f'episode{episode}'
            numeric = np.load(folder / 'joints.npz', allow_pickle=False)
            self.arrays[name] = (np.load(folder / 'rgb.npy', mmap_mode='r'),
                                 numeric['states'], numeric['actions'])
        rgb, states, actions = self.arrays[name]
        offsets = t + np.arange(self.horizon)
        future = np.minimum(offsets, len(actions) - 1)
        s = self.stats
        state = (states[t] - s['s_low']) / (np.array(s['s_high']) - s['s_low'] + 1e-6) * 2 - 1
        action = (actions[future] - s['a_low']) / (np.array(s['a_high']) - s['a_low'] + 1e-6) * 2 - 1
        mask = pack(np.ones((self.horizon, 14), dtype=np.float32))
        mask[offsets >= len(actions)] = 0
        prompts = self.manifest['tasks'][task]['episodes'][str(episode)]['seen']
        text = prompts[int(seed) % len(prompts)]
        return {'rgb': augment_rgb(rgb[t], seed) if augmentation else np.array(rgb[t], copy=True),
                'state': pack(state), 'actions': pack(action), 'joint_mask': mask,
                'instruction': text, 'task': task, 'seed': seed}


class WindowDataset:
    def __init__(self, root, manifest, stats, horizon):
        self.store = EpisodeStore(root, manifest, stats, horizon)

    def __getitem__(self, key):
        return self.store.sample(key)


class UpdateSampler:
    def __init__(self, manifest, cfg, first_step, micro):
        self.manifest, self.cfg, self.first_step, self.micro = manifest, cfg, first_step, micro

    def __iter__(self):
        for step in range(self.first_step, self.cfg['steps'] + 1):
            chosen = selections(self.manifest, self.cfg, step)
            for start in range(0, len(chosen), self.micro):
                yield chosen[start:start + self.micro]

    def __len__(self):
        return max(0, self.cfg['steps'] - self.first_step + 1) * self.cfg['effective_batch'] // self.micro


class BatchCollator:
    def __init__(self, processor_dir, max_length):
        self.processor_dir, self.max_length = str(processor_dir), max_length
        self.processor = None

    def __call__(self, samples):
        import numpy as np
        import torch
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.processor_dir, local_files_only=True)
        p = self.processor
        images = p.image_processor(images=[v for s in samples for v in s['rgb']],
                                   do_resize=False, return_tensors='pt')
        prompts = [p.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': s['instruction']}], tokenize=False,
            add_generation_prompt=False) for s in samples]
        # 不静默截断任务指令：超长时明确提示增加 max_text_length。
        encoded = p.tokenizer(prompts, add_special_tokens=False, padding='max_length',
                              truncation=False, max_length=self.max_length, return_tensors='pt')
        if encoded['input_ids'].shape[1] > self.max_length:
            raise ValueError('官方指令超出 max_text_length，请在配置中增加长度')
        b = len(samples)
        pixels, grid = images['pixel_values'], images['image_grid_thw']
        assert torch.all(grid == grid[0])
        return {'images': pixels.reshape(b, 3, -1, pixels.shape[-1]),
                'image_grid_thw': grid.reshape(b, 3, 3),
                'img_masks': torch.ones(b, 3, dtype=torch.bool),
                'lang_tokens': encoded['input_ids'], 'lang_masks': encoded['attention_mask'].bool(),
                **{k: torch.from_numpy(np.stack([s[k] for s in samples]))
                   for k in ['state', 'actions', 'joint_mask']}}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

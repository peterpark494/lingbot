import fcntl, json, os, signal, subprocess, time
from pathlib import Path
from datetime import datetime, timezone

# 路径和子进程环境不依赖 Notebook 内核中上一次留下的变量。
WORK = Path('/content/lingbot_full10')
SCRIPTS = WORK / 'scripts'
CONFIG = WORK / 'config.json'
PY = Path('/content/lingbot_full10_env/bin/python')
missing = [str(p) for p in [CONFIG, PY, SCRIPTS / 'colab10_train.py', WORK / 'data_manifest.json'] if not p.is_file()]
if missing:
    raise RuntimeError('当前是尚未准备完成的新运行时。需要先恢复 Drive、独立环境和本地数据缓存；'
                       '不会从头启动新实验。缺少：' + ', '.join(missing))
RUN_ENV = os.environ.copy()
RUN_ENV.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
RUN_ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 只改 Notebook 启动方式，不改训练程序、超参数或 checkpoint。
console_path = WORK / 'training_console.log'
console_path.touch(exist_ok=True)
run_dir = Path(json.loads(CONFIG.read_text())['save_root']) / json.loads(CONFIG.read_text())['run_name']

def training_locked():
    with (WORK / 'training.lock').open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe, fcntl.LOCK_UN)
        return False

def brief_progress():
    pointer = run_dir / 'latest.json'
    saved = json.loads(pointer.read_text())['step'] if pointer.exists() else '尚无'
    pending = [(p.name, round(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1024**3, 2))
               for p in run_dir.glob('*.incomplete')]
    print('最近完整断点：', saved, '；正在写入的目录及 GiB：', pending, flush=True)
    for cache in (WORK / 'resume_cache').glob('*.copying-*'):
        files = [p for p in cache.rglob('*') if p.is_file()]
        print('本地恢复文件准备中：', len(files), '个文件，',
              round(sum(p.stat().st_size for p in files) / 1024**3, 2), 'GiB', flush=True)
    if (WORK / 'runtime_restore_request.json').exists():
        count = len(list((WORK / 'prepared').glob('*/episode*/complete.json')))
        print('本地示范缓存：', count, '/500；此阶段为恢复准备，尚未更新参数。', flush=True)
        recovery_log = WORK / 'runtime_restore_console.log'
        if recovery_log.exists():
            print(''.join(recovery_log.read_text(errors='replace').splitlines(keepends=True)[-3:]), end='', flush=True)

def follow(reader):
    text = reader.read()
    if text:
        print(text, end='', flush=True)

print('训练控制台日志：', console_path, flush=True)
if training_locked():
    print('已有任务持有运行锁：等待训练/保存/本地断点准备结束，不启动第二个训练进程。', flush=True)
    print('若正在保存，中断保存可能丢失尚未发布的进度；请勿删除运行时。', flush=True)
    with console_path.open('r', encoding='utf-8', errors='replace') as reader:
        reader.seek(0, 2)
        last_report = 0
        while training_locked():
            follow(reader)
            if time.monotonic() - last_report >= 60:
                brief_progress()
                last_report = time.monotonic()
            time.sleep(2)
        follow(reader)
    local_log = WORK / 'train.jsonl'
    records = local_log.read_text().splitlines() if local_log.exists() else []
    last_update = json.loads(records[-1])['step'] if records else 0
    pointer = run_dir / 'latest.json'
    saved_step = json.loads(pointer.read_text())['step'] if pointer.exists() else 0
    if saved_step < last_update:
        raise RuntimeError(f'原进程退出但断点只保存到 {saved_step}，内存更新曾到 {last_update}；停止自动启动，请检查保存错误。')
    print('原进程已释放训练锁；现在按原配置读取最新完整断点继续。', flush=True)

if (WORK / 'runtime_restore_request.json').exists() and not (WORK / 'runtime_restore_ready.json').exists():
    raise RuntimeError('新会话恢复准备尚未成功完成；未启动训练。请查看终端中的恢复错误。')

# 训练进程直接写日志文件；Notebook 只负责读取，避免界面断开后管道丢失。
with console_path.open('a', encoding='utf-8', buffering=1) as output:
    output.write('\n=== ' + datetime.now(timezone.utc).isoformat() + ' ===\n')
    with console_path.open('r', encoding='utf-8', errors='replace') as reader:
        reader.seek(0, 2)
        process = subprocess.Popen(
            [str(PY), '-u', str(SCRIPTS / 'colab10_train.py'), '--config', str(CONFIG)],
            env=RUN_ENV, start_new_session=True, stdout=output, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                follow(reader)
                time.sleep(1)
            follow(reader)
        except KeyboardInterrupt:
            print('已请求中断；等待完整更新保存到 Drive。请勿反复点击停止。', flush=True)
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            last_report = 0
            while process.poll() is None:
                follow(reader)
                if time.monotonic() - last_report >= 60:
                    brief_progress()
                    last_report = time.monotonic()
                time.sleep(1)
            follow(reader)
        return_code = process.wait()
if return_code:
    raise RuntimeError(f'训练进程退出，返回码 {return_code}；具体错误见上方日志。')

import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/wowow2/tinyvlm.git"
SRC = "/tmp/tinyvlm"
OUT = "/kaggle/working"


def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"devices={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        print(f"nvidia-smi: {(smi.stdout or smi.stderr).strip()}", flush=True)
    except Exception as e:
        print(f"nvidia-smi failed: {type(e).__name__}: {e}", flush=True)

    try:
        import yaml  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])

    if os.path.exists(SRC):
        shutil.rmtree(SRC)
    try:
        run(["git", "clone", "--depth", "1", REPO, SRC])
    except subprocess.CalledProcessError:
        print("GIT CLONE FAILED (no internet?)", flush=True)
        raise

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(SRC, "src")

    run([sys.executable, "scripts/download_data.py"], cwd=SRC, env=env)

    t0 = time.perf_counter()
    run([sys.executable, "scripts/train.py"], cwd=SRC, env=env)
    print(f"\ntraining + sample wall time: {time.perf_counter() - t0:.1f}s", flush=True)

    ckpt = os.path.join(SRC, "checkpoints", "tiny.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    dst = os.path.join(OUT, "tiny.pt")
    shutil.copy2(ckpt, dst)
    print(f"saved {dst} ({os.path.getsize(dst)} bytes)", flush=True)
    print(f"working dir: {sorted(os.listdir(OUT))}", flush=True)


if __name__ == "__main__":
    main()

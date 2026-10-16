# -*- coding: utf-8 -*-
"""为 GitHub Actions 及本地构建准备 embedded_torch 运行时目录。"""
import sys
import os
import shutil
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

def prepare(variant: str):
    dst_name = "embedded_torch" if variant == "cpu" else "embedded_torch_cuda"
    dst = os.path.join(HERE, dst_name)
    print(f"[prepare] 准备变体: {variant} -> {dst_name}/")

    try:
        import torch
        import torchaudio
    except ImportError:
        print("[ERROR] 未安装 torch / torchaudio")
        sys.exit(1)

    sp = os.path.dirname(os.path.dirname(torch.__file__))
    print(f"[info] site-packages: {sp}")
    print(f"[info] torch: {torch.__version__}, cuda: {getattr(torch.version, 'cuda', None)}")

    pkgs = ["torch", "torchaudio", "torio", "functorch", "torchgen"]
    dev_exts = (".lib", ".exp", ".obj", ".o", ".a", ".pdb", ".h", ".hpp", ".cuh", ".cu", ".cpp", ".c", ".cc", ".pyi")
    skip_dirs = ("__pycache__", "tests", "test", "examples", "benchmarks", "docs", ".git", "build", "dist", "include")

    def _ignore(src, names):
        out = []
        for n in names:
            ln = n.lower()
            if ln in skip_dirs:
                out.append(n)
            elif ln.endswith(dev_exts):
                out.append(n)
        return out

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    for p in pkgs:
        src = os.path.join(sp, p)
        if not os.path.isdir(src):
            continue
        shutil.copytree(src, os.path.join(dst, p), ignore=_ignore)
        for di in glob.glob(os.path.join(sp, p + "-*.dist-info")):
            if os.path.isdir(di):
                shutil.copytree(di, os.path.join(dst, os.path.basename(di)), ignore=_ignore)

    bin_dir = os.path.join(dst, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    vc_names = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll")
    for n in vc_names:
        src = os.path.join(sys.prefix, n)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bin_dir, n))

    print(f"[OK] {dst_name}/ 准备完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["cpu", "gpu"], required=True)
    args = parser.parse_args()
    prepare(args.variant)

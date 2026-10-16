"""扫描 embedded torch 启动所需的全部标准库模块，输出可直接用于 --hidden-import 的列表。

原理：在 import embedded_torch 期间拦截 __builtins__.__import__，记录所有顶层属于
sys.stdlib_module_names 的模块（含多级前缀），确保 PyInstaller 冻结后这些模块都在包内。

用法:
  python _collect_stdlib.py                       # 默认扫描 embedded_torch/ -> _stdlib_imports.txt
  python _collect_stdlib.py --target embedded_torch_cuda --out _stdlib_imports_gpu.txt
"""
import builtins as _b
import sys as _s
import os as _os

HERE = _os.path.dirname(_os.path.abspath(__file__))
emb = _os.path.join(HERE, "embedded_torch")
OUT = _os.path.join(HERE, "_stdlib_imports.txt")

# 支持 --target <dir> --out <file> 指定扫描目标（GPU 版用 embedded_torch_cuda）
_args = _s.argv[1:]
if "--target" in _args:
    emb = _os.path.join(HERE, _args[_args.index("--target") + 1])
if "--out" in _args:
    OUT = _os.path.join(HERE, _args[_args.index("--out") + 1])

_std_names = set(_s.stdlib_module_names)
_collected = set()
_orig = _b.__import__


def _rec(name, *a, **k):
    if isinstance(name, str):
        top = name.split(".")[0]
        if top in _std_names:
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                _collected.add(".".join(parts[:i]))
    return _orig(name, *a, **k)


_b.__import__ = _rec
try:
    if emb not in _s.path:
        _s.path.insert(0, emb)
    import torch  # noqa: F401
finally:
    _b.__import__ = _orig

mods = sorted(_collected)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(mods))
print("COLLECTED_STDLIB_COUNT=", len(mods))
print("OUTPUT=", OUT)
print("\n".join(mods))

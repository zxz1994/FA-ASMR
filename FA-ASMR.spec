# -*- mode: python ; coding: utf-8 -*-
"""FA-ASMR V2 PyInstaller 规格（变体化模板）。

本文件是**模板**，由 `_build_exe.py` 派生变体化临时 spec 后使用，
不建议直接调用 `pyinstaller FA-ASMR.spec`。占位符（由构建脚本替换）:

    @PROJECT_DIR@      -> 项目根绝对路径（spec 被派生到 build_run/ 下，必须锚定）
    @EMBEDDED_TORCH@   -> embedded_torch（CPU 版）/ embedded_torch_cuda（GPU 版）
    @APP_NAME@         -> FA-ASMR（CPU 版）/ FA-ASMR-GPU（GPU 版）

包内目录名恒为 embedded_torch（与 torch_bootstrap._try_embedded_torch() 约定一致），
CPU/GPU 差异仅体现在其内容，由 _internal/variant.txt 在运行时标记。

── 收集策略（依据实际运行链路实测，非静态推断）────────────────────────────
必须 collect_all（含非 .py 数据文件 / 二进制 / 词典）:
    yomikata          dBert 权重 dbert-artifacts/（约 433 MB）+ config/heteronyms.json
    transformers      模块级被 yomikata 依赖，且其内部字典/配置需随包
    tokenizers        transformers 分词器原生后端
    huggingface_hub   transformers 依赖
    safetensors       权重加载后端
    numba / llvmlite  align_trellis 的 JIT 必需（缺失即 RuntimeError）
    janome            haruraw2norm._fix_ha_he 的助词修正
    unidic_lite       fugashi 的 UniDic 词典
    fugashi / pykakasi / tkinterdnd2 / PIL / _soundfile_data

必须排除（体积大且链路不用）:
    sudachipy / sudachidict_core  仅注释提及，代码零引用（省约 221 MB）
    matplotlib / pandas / cv2 / torchvision / onnxruntime / onnx / sklearn
    nltk / pyphen / pypinyin      中英文场景的可选依赖，haruraw2norm 内 try/except 包裹
    torch / torchaudio            由 embedded_torch/ 在运行时注入，不走 PyInstaller
"""
import os
from PyInstaller.utils.hooks import collect_all

# ── 路径基准 ──────────────────────────────────────────────────────────────
# PyInstaller 执行 spec 时以「spec 文件所在目录」为相对路径基准，而本模板会被
# _build_exe.py 派生到 build_run/ 下，因此所有路径必须锚定到项目根。
# @PROJECT_DIR@ 由 _build_exe.py 注入（最可靠，不依赖 SPECPATH 是否存在）。
PROJECT_DIR = r"@PROJECT_DIR@"


def _abs(*parts):
    return os.path.join(PROJECT_DIR, *parts)


APP_NAME = "@APP_NAME@"
EMBEDDED_TORCH = "@EMBEDDED_TORCH@"

datas = [
    (_abs('assets'), 'assets'),
    (_abs('furigana'), 'furigana'),
    # lora/ 打进 _internal/lora/：align.py 的 DEFAULT_LORA_PATH 以 __file__ 定位
    # （frozen 下即 _internal/），因此必须位于 _internal 内，而非 exe 同级。
    (_abs('lora'), 'lora'),
    # 内置 torch：源目录名随变体变化，包内目录名恒为 embedded_torch
    # （与 torch_bootstrap._try_embedded_torch() 的约定一致）。
    (_abs(EMBEDDED_TORCH), 'embedded_torch'),
]
binaries = []

# ─────────────────────────────────────────────────────────────────────────
# IMPORTANT —— 以下是实测确认的运行时硬依赖，误排除会让打包版启动即崩：
#
#   * transformers：yomikata/dbert.py 在**模块级**执行
#       from transformers import (AutoModelForTokenClassification,
#                                 BertJapaneseTokenizer, ..., Trainer, ...)
#     缺它时注音阶段抛 RuntimeError("yomikata dBert 加载失败") 并终止。
#     原 spec 曾把它列入 excludes —— 那是错的。
#
#   * peft：transformers/trainer.py:226 有 `from peft import PeftModel`。
#     排除 peft 会让 transformers.trainer 导入失败，进而连带 dbert 导入失败。
#     （本项目自研最小 LoRA，确实不用 peft，但 transformers 会主动 import 它。）
#
#   * scipy：transformers.generation.utils 依赖 scipy.sparse。
#     排除 scipy 会让 transformers.models.auto.modeling_auto 导入失败。
#
# 综上：excludes 里**不可**出现 transformers / peft / scipy / tokenizers /
#       huggingface_hub / safetensors。它们约 130 MB，但属注音链必需成本。
# ─────────────────────────────────────────────────────────────────────────

hiddenimports = [
    # 本项目模块（部分为运行期延迟导入，静态分析不一定收录）
    'soundfile', 'jaconv',
    'align', 'align_acoustic', 'align_model', 'align_utils', 'align_post',
    'align_trellis', 'haruraw2norm', 'lrc_end_extend', 'resegment',
    'advanced_config', 'torch_bootstrap',
    'inline_leak_clean', 'merge_short', 'ono_filter', 'ono_base', 'ono_block_corpus',
    'clause', 'np_compat',

    # ── 必须显式声明：PyInstaller 的依赖分析**漏收**这两者 ──
    # 实测（屏蔽即崩，报 "Unable to import required dependency dateutil"）：
    #   dateutil  —— transformers.trainer 声明的必需依赖，走 importlib.metadata
    #                检查后动态导入，静态分析看不见。
    #   narwhals  —— 由 sklearn.utils.validation 引入（narwhals.stable.v2）；
    #                transformers 链路上会触达，漏收同样导致注音终止。
    # 二者体积合计约 5 MB，代价可忽略。
    'dateutil', 'dateutil.parser', 'dateutil.tz',
    'narwhals',
]

# 标准库清单（_collect_stdlib.py 扫描 embedded_torch 启动期导入生成）
_STDLIB_FILE = "_stdlib_imports_gpu.txt" if EMBEDDED_TORCH.endswith("cuda") else "_stdlib_imports.txt"
try:
    _std_path = _abs(_STDLIB_FILE)
    with open(_std_path, encoding='utf-8') as _f:
        _stdlib = [ln.strip() for ln in _f if ln.strip()]
    hiddenimports += _stdlib
except Exception as _e:
    print("[spec] 警告: 无法读取 %s（%s）；标准库依赖可能不完整。" % (_std_path, _e))

# 含数据文件 / 原生扩展的第三方包
#
# speach 必须列入：yomikata.dbert / yomikata.utils 在**模块级**执行
#   from speach.ttlig import RubyFrag, RubyToken
# 且 speach.ttlig 又有 `import chirptext`（chirptext.__init__ 再拉
# puchikarui），PyInstaller 静态分析看不穿这层间接导入，漏收即报
#   ModuleNotFoundError: No module named 'speach'
# 导致注音阶段直接终止（此前云端构建就栽在这里）。
for _pkg in ('tkinterdnd2', 'pykakasi', 'janome', 'fugashi', 'unidic_lite',
             'yomikata', 'transformers', 'tokenizers', 'huggingface_hub',
             'safetensors', 'numba', 'llvmlite', 'PIL', '_soundfile_data',
             'speach', 'chirptext', 'puchikarui', 'rich', 'pynvml'):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# ─────────────────────────────────────────────────────────────────────────
# 多余包剔除（**必须在 Analysis 之后执行**，见下方说明）
#
# 为什么不能只靠 excludes：
#   1) collect_all('safetensors') 会连 safetensors.paddle / .tensorflow / .flax
#      一并收进 hiddenimports，Analysis 依赖图据此拉入 paddle(218 MB)、spacy、
#      pyarrow、polars 整棵生态；
#   2) hook-torch 的 collect_dynamic_libs("torch") 会把 site-packages 里那份
#      **CUDA 版 torch 的 lib/**（约 2.4 GB DLL）也收进来，而本项目运行期用的是
#      embedded_torch/ 内的 CPU torch —— 两者重复且 ABI 冲突。
#   excludes 只作用于「模块导入图」，管不住 hook 的 collect_* 结果，
#   因此只能在 Analysis 完成后对 a.pure / a.binaries / a.datas 统一过滤。
_BLOCKED = frozenset([
    # ── 实测可安全剔除（逐个单包剔除后，注音与对齐链路均正常）──
    # 注意 _polars_runtime_* 是独立顶层包（polars 的 native 运行时），
    # 名字不以 "polars" 开头，需显式列出。
    'paddle', 'spacy', 'polars',
    '_polars_runtime_32', '_polars_runtime_64',
    'tensorflow', 'flax', 'jax', 'jaxlib', 'ml_dtypes',
    'matplotlib', 'cv2', 'torchvision',
    'onnxruntime', 'sklearn', 'scikit_learn',
    'timm', 'trimesh', 'networkx', 'librosa',
    'optuna', 'sentencepiece', 'tensorboardX',
    # 日语分词：sudachi 系代码零引用（仅注释提及），省约 221 MB
    'sudachipy', 'sudachidict_core', 'sudachidict_full', 'sudachidict_small',
    # 中英文场景可选依赖（haruraw2norm 内 try/except 包裹，日文不需要）
    'nltk', 'pyphen', 'pypinyin',
    # 训练框架
    'lightning', 'pytorch_lightning',

    # ── 必须「三元同剔」──
    # transformers.trainer 会按包元数据判断 datasets 可用性；只要 datasets 在，
    # 它就会硬 import pandas 与 pyarrow。单独剔除其中任一个都会让
    # transformers.trainer 导入失败，进而 dbert 加载失败、注音终止。
    # 三者同时剔除后 transformers 判定为不可用并跳过，链路正常（已实测）。
    'datasets', 'pandas', 'pyarrow',

    # ── 内置 torch 由 torch_bootstrap 在运行期从 embedded_torch/ 注入，不走冻结 ──
    # 注意：torio/torchgen 是 yomikata 依赖链在 **开发机** 上间接引入的，
    # 打包版由 embedded_torch 提供同名包，故 site-packages 侧应剔除。
    'torch', 'torchaudio', 'torchgen', 'functorch', 'torio', 'torchtext',
])

# 实测【必须保留】的网络/数学库（勿加入 _BLOCKED，否则注音链会断）:
#   sympy        —— transformers.trainer 依赖
#   accelerate   —— transformers.trainer 依赖
#   onnx         —— transformers.models.bloom 依赖
#   tokenizers / huggingface_hub / safetensors / transformers / yomikata / numba / llvmlite
# 这些虽大，但属运行时必需成本。

# torch 的 CUDA 运行时 DLL（embedded_torch 为 CPU 版，不需要）
_BLOCKED_DLL_PREFIX = ('torch_cuda', 'cublas', 'cudnn', 'cufft', 'cusparse',
                       'cusolver', 'cudart', 'nvrtc', 'nvToolsExt', 'cupti')


def _blocked_entry(name, path):
    """判断一条 TOC 记录是否应剔除。

    name 为运行期名（dest / 模块全名），path 为构建机上的源路径。

    注意 name 的形态随 TOC 而异：
      a.pure      -> Python 模块全名，用 '.' 分隔（如 paddle._typing.libs）
      a.binaries  -> 相对路径，用 '/' 分隔
      a.datas     -> 相对路径，含 '.' 的文件名（如 x.py）

    ⚠️ 历史教训（曾导致打包版注音链崩溃，7 个任务失败 6 个）：
    **绝不能**对模块名做「任意分量命中黑名单」的匹配。`_BLOCKED` 里的包名
    （torch / paddle / tensorflow / flax / jax ...）同时也是合法模块的中间分量，
    如 `safetensors.torch` 的中间分量即 `torch`。按分量拦会连带删掉它，
    而 `transformers.pytorch_utils` 在模块级导入 `safetensors.torch`，
    于是 yomikata.dbert 加载失败、注音终止。

    正确做法：只按【顶层】名判定；源路径只认 site-packages 下的【直接子目录名】。
    """
    n = str(name).replace('\\', '/')
    p = str(path).replace('\\', '/')

    # 内置运行时：**以目标位置(dest)判定豁免范围**。
    # CPU 版源目录为 embedded_torch/，GPU 版为 embedded_torch_cuda/（包内仍叫
    # embedded_torch）。包内 embedded_torch/ 这棵树是运行期真正要用的，
    # 包括 GPU 版 lib/ 下的 cudnn/cublas/cudart 等（它们恰好命中
    # _BLOCKED_DLL_PREFIX，若按名字拦截会把 GPU 版自身的 CUDA 运行时删掉）。
    if n.startswith('embedded_torch'):
        return False

    # 源在内置树、但 dest 不在该树下 —— 这是 PyInstaller 的 bindepend 阶段
    # 解析 .pyd 导入表后，把依赖 DLL 另存到 _internal/ 顶层产生的**重复副本**
    # （实测与 embedded_torch/ 内原件逐比特相同，最多可占 GPU 版体积的 30%）。
    # torch_bootstrap 已把 embedded_torch/torch/lib 与 embedded_torch/bin
    # 加入 DLL 搜索目录，顶层副本纯属冗余，剔除。
    if '/embedded_torch/' in p or '/embedded_torch_cuda/' in p:
        return True

    # ① 顶层名判定：用 '.' 与 '/' 统一取首段。
    #    覆盖 a.pure 的 "paddle.foo"、a.datas 的 "paddle/foo"、
    #    dist-info 形态 "polars-1.43.2.dist-info"。
    top = n.replace('.', '/').split('/')[0].lower()
    if top in _BLOCKED or top.split('-')[0] in _BLOCKED:
        return True

    # ② 源路径判定：只认 site-packages 下的**直接子目录名**。
    #    ⚠️ 绝不做全路径分量扫描：`_BLOCKED` 里放的包名（torch / paddle /
    #    tensorflow / flax / jax ...）同时也是合法模块的中间分量，例如
    #    `safetensors.torch` 的中间分量就是 `torch`。按分量拦会把
    #    `safetensors.torch` 一并删掉，而 transformers.pytorch_utils 在模块级
    #    `from safetensors.torch import storage_ptr, storage_size`，于是
    #    transformers.generation.utils -> models.auto.modeling_auto ->
    #    yomikata.dbert 整条链崩溃（打包版注音直接失败）。
    #    同理 ".../sympy/plotting/backends/matplotlibbackend/matplotlib.py"
    #    也会被误删。
    if '/site-packages/' in p:
        tail = p.split('/site-packages/', 1)[1]
        first = tail.split('/')[0].lower()
        if first in _BLOCKED or first.split('-')[0] in _BLOCKED:
            return True

    # CUDA 运行时 DLL（仅 site-packages 侧；内置树的已在上面豁免）。
    # 限定二进制扩展名，避免误伤同名的 .py（如 numba.cuda.cudadrv.nvrtc）。
    base = p.rsplit('/', 1)[-1]
    if base.endswith(('.dll', '.pyd', '.so')):
        if any(base.startswith(x) for x in _BLOCKED_DLL_PREFIX):
            return True
    return False


def _filter(toc, label):
    kept = [e for e in toc if not _blocked_entry(e[0], e[1])]
    if len(kept) != len(toc):
        print("[spec] %s: %d -> %d" % (label, len(toc), len(kept)))
    return kept


a = Analysis(
    [_abs('FA-ASMR_GUI.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 体积大且链路不使用
        'matplotlib', 'pandas', 'cv2', 'torchvision',
        'onnxruntime', 'onnx', 'sklearn', 'scikit_learn',
        # 日语分词：sudachi 系代码零引用（仅注释提及）
        'sudachipy', 'sudachidict_core', 'sudachidict_full', 'sudachidict_small',
        # 中英文场景可选依赖（haruraw2norm 内 try/except 包裹，日文不需要）
        'nltk', 'pyphen', 'pypinyin',
        # 内置 torch 由 torch_bootstrap 在运行期从 embedded_torch/ 注入，不走冻结
        'torch', 'torchaudio',
        # 避免与 embedded torch 的 ABI 混淆
        'scipy._external.array_api_compat.torch',
        'sklearn.externals.array_api_compat.torch',
    ],
    noarchive=False,
    optimize=0,
)
# ── 过滤（必须在 Analysis 之后，理由见上方注释）──
a.pure = _filter(a.pure, "a.pure")
a.binaries = _filter(a.binaries, "a.binaries")
a.datas = _filter(a.datas, "a.datas")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

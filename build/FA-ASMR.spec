# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('tasks', 'tasks'), ('furigana', 'furigana'), ('lora', 'lora'), ('embedded_torch', 'embedded_torch'), ('ono_table.json', '.'), ('fa_asmr_settings.json', '.')]
binaries = []
hiddenimports = ['soundfile', 'scipy', 'jaconv', 'align_model', 'align_utils', 'align_post', 'advanced_config', 'timeit', '__future__', '_asyncio', '_codecs', '_collections', '_compat_pickle', '_compression', '_contextvars', '_csv', '_datetime', '_functools', '_heapq', '_imp', '_io', '_multiprocessing', '_operator', '_overlapped', '_pickle', '_pyio', '_queue', '_socket', '_ssl', '_thread', '_uuid', '_warnings', '_weakrefset', '_winapi', 'abc', 'argparse', 'ast', 'asyncio', 'asyncio.base_futures', 'asyncio.base_tasks', 'asyncio.coroutines', 'asyncio.events', 'asyncio.exceptions', 'atexit', 'base64', 'bdb', 'binascii', 'bisect', 'builtins', 'bz2', 'calendar', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'collections.abc', 'concurrent', 'concurrent.futures', 'concurrent.futures._base', 'contextlib', 'contextvars', 'copy', 'copyreg', 'csv', 'ctypes', 'ctypes.util', 'ctypes.wintypes', 'dataclasses', 'datetime', 'difflib', 'dis', 'email', 'email._encoded_words', 'email._parseaddr', 'email._policybase', 'email.base64mime', 'email.charset', 'email.encoders', 'email.errors', 'email.feedparser', 'email.iterators', 'email.message', 'email.parser', 'email.quoprimime', 'email.utils', 'enum', 'errno', 'fnmatch', 'functools', 'gc', 'gettext', 'glob', 'grp', 'gzip', 'hashlib', 'heapq', 'http', 'http.client', 'importlib', 'importlib.abc', 'importlib.machinery', 'importlib.metadata', 'importlib.util', 'inspect', 'io', 'itertools', 'json', 'keyword', 'linecache', 'locale', 'logging', 'lzma', 'marshal', 'math', 'msvcrt', 'multiprocessing', 'multiprocessing.connection', 'multiprocessing.reduction', 'multiprocessing.resource_sharer', 'multiprocessing.util', 'nt', 'nturl2path', 'numbers', 'operator', 'os', 'os.path', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pkgutil', 'platform', 'posixpath', 'pprint', 'pwd', 'queue', 'quopri', 'random', 're', 'reprlib', 'selectors', 'shutil', 'signal', 'socket', 'ssl', 'stat', 'string', 'struct', 'subprocess', 'sys', 'sysconfig', 'tarfile', 'tempfile', 'textwrap', 'threading', 'time', 'timeit', 'tokenize', 'traceback', 'types', 'typing', 'unicodedata', 'unittest', 'unittest.mock', 'unittest.util', 'urllib', 'urllib.error', 'urllib.parse', 'urllib.request', 'urllib.response', 'uuid', 'warnings', 'weakref', 'winreg', 'zipfile', 'zipimport', 'zlib']
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pykakasi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('janome')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fugashi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sudachidict_core')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sudachipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['FA-ASMR_GUI.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'PIL', 'cv2', 'nltk', 'pyphen', 'pypinyin', 'torch', 'torchaudio', 'peft', 'transformers', 'tokenizers', 'huggingface_hub', 'torchvision', 'onnxruntime', 'onnx', 'scipy._external.array_api_compat.torch', 'sklearn.externals.array_api_compat.torch'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FA-ASMR',
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
    name='FA-ASMR',
)

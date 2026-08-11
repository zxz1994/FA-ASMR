# PyInstaller 运行时钩子：修复 scipy.stats 懒加载导致的 NameError
# 在程序入口之前强制导入 scipy.stats，避免 bytecode 优化破坏闭包引用

import scipy
import scipy.stats
# 强制触发 _distn_infrastructure 加载（修复 'obj' NameError）
try:
    from scipy.stats import norm
except Exception:
    pass

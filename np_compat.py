# -*- coding: utf-8 -*-
"""NumPy 2.x 别名兼容补丁（全项目唯一实现）。

背景
----
NumPy 2.0 移除了 ``np.long`` / ``np.ulong`` / ``np.float`` / ``np.bool``
这些别名，但部分依赖链仍在引用它们：

- ``yomikata`` / ``transformers`` 的老代码路径
- 打包进来的第三方包

缺失时表现为 ``AttributeError: module 'numpy' has no attribute 'long'``，
且发生在导入期——不补则模块直接加载失败。

为什么单独成文件
----------------
这段补丁原先在三个文件里各写了一份完全相同的副本：

- ``FA-ASMR_GUI.py``（导入期）
- ``haruraw2norm.py``（导入期）
- ``furigana/fa_asmr_converter.py``（**函数内**，加载 yomikata 时惰性执行）

三份字面完全相同，属真实重复。提取后三处改为调用 ``apply()``。

幂等
----
``apply()`` 可安全调用任意多次：每次都先检查 ``np.__dict__``，已存在则跳过。
这一点必须保持——``fa_asmr_converter`` 是在惰性加载路径里调用它的，
与另外两处的导入期调用会叠加。
"""


def apply() -> int:
    """补齐 NumPy 2.x 移除的标量类型别名。返回实际补上的个数。"""
    import numpy as np

    patched = 0
    for name, value in (("long", int), ("ulong", int),
                        ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
            patched += 1
    return patched

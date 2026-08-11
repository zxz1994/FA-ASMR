"""Stub hook: 覆盖 PyInstaller 内置的 nltk 钩子，阻止生成 pyi_rth_nltk.py
FA-ASMR 不需要 nltk，它的大部分导入链会触发 scipy NameError 崩溃。"""

# 不生成任何运行时钩子，不收集额外数据

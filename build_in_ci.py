# -*- coding: utf-8 -*-
"""GitHub Actions CI 环境构建与打包脚本。"""
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# CI 控制台默认编码为 cp1252/cp932，中文日志会抛 UnicodeEncodeError。
# 必须在任何输出之前把标准流切到 UTF-8，并让子进程继承同一编码。
os.environ["PYTHONIOENCODING"] = "utf-8"
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent

def _find_7z() -> str:
    """定位 7-Zip（GitHub 托管 runner 与本地安装路径不同）。"""
    for cand in (r"C:\Program Files\7-Zip\7z.exe",
                 r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if os.path.isfile(cand):
            return cand
    found = shutil.which("7z") or shutil.which("7z.exe")
    if found:
        return found
    print("[FAIL] 未找到 7z.exe，无法压缩产物。", flush=True)
    sys.exit(1)

SEVEN_ZIP = _find_7z()
BUILD_ROOT = HERE / "build_run"
DIST_ROOT = HERE / "dist"
ARTIFACTS_DIR = HERE / "artifacts"
MODEL_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"

def log(msg):
    print(f"[CI-BUILD] {msg}", flush=True)

def fail(msg):
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(1)

def _site_packages() -> Path:
    import site
    for p in site.getsitepackages():
        if "site-packages" in p:
            return Path(p)
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"])


def ensure_dbert_artifacts() -> Path:
    """确保 yomikata dBert 权重就绪。

    dbert-artifacts/ 约 423 MB，**不在 PyPI wheel 内**（RECORD 中无该目录），
    官方只提供 `python -m yomikata download` 单独拉取。
    缺失时 dbert.py 的 load() 会抛异常，注音阶段直接终止。
    """
    dest = _site_packages() / "yomikata" / "dbert-artifacts"
    weight = dest / "pytorch_model.bin"

    if weight.is_file() and weight.stat().st_size > 400_000_000:
        log(f"dBert 权重已就绪: {dest} ({weight.stat().st_size / 1e6:.1f} MB)")
        return dest

    log("正在下载 yomikata dBert 权重（约 423 MB）...")
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, "-m", "yomikata", "download"],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if res.returncode != 0 or not weight.is_file():
        fail("dBert 权重下载失败。离线构建时请预先把 dbert-artifacts/ "
             "放入 site-packages/yomikata/ 下。")

    log(f"dBert 权重就绪，耗时 {time.time()-t0:.1f}s，"
        f"大小 {weight.stat().st_size / 1e6:.1f} MB")
    return dest


def ensure_model():
    model_path = HERE / "models" / "hub" / "checkpoints" / "model.pt"
    if model_path.is_file() and model_path.stat().st_size > 1000000000:
        log(f"本地模型已就绪: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"正在从 Meta 官方 CDN 下载 MMS-FA model.pt ...")
    t0 = time.time()

    res = subprocess.run(["curl.exe", "-L", "-o", str(model_path), MODEL_URL, "--progress-bar"])
    if res.returncode != 0 or not model_path.is_file():
        fail("下载 MMS-FA model.pt 失败！")

    log(f"模型下载完成，耗时 {time.time()-t0:.1f}s，大小: {model_path.stat().st_size / 1e6:.1f} MB")
    return model_path


def verify_annotator():
    """构建前实跑一次真实注音链，确保依赖与权重齐全。

    只做启动冒烟测试无法发现注音链缺件（曾漏 speach 却仍能生成 exe），
    因此这里直接调用生产代码路径。
    """
    log("构建前验证注音链路（真实调用 RubyAnnotator）...")
    code = (
        "import sys, os;"
        f"sys.path.insert(0, os.path.join(r'{HERE}', 'furigana'));"
        f"sys.path.insert(0, r'{HERE}');"
        "from fa_asmr_converter import RubyAnnotator;"
        "ann = RubyAnnotator(verbose=False, strip_inline_readings=True);"
        "out = ann.annotate('\u6c96\u7e04\u306e\u30db\u30c6\u30eb\u3067\u3001"
        "\u4e8c\u4eba\u304d\u308a\u306e\u591c\u3092\u904e\u3054\u3057\u307e\u3057\u3087\u3046\u306d\u3002');"
        "print('ANNOTATOR_OK', out[:40])"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if res.returncode != 0:
        fail("注音链路验证失败：依赖或 dBert 权重不完整，构建已中止。")
    log("注音链路验证通过。")

def main(variant: str):
    log(f"==================================================")
    log(f"  开始在 CI 虚拟机构建 {variant.upper()} 版本")
    log(f"==================================================")
    
    app_name = "FA-ASMR" if variant == "cpu" else "FA-ASMR-GPU"
    emb_src = "embedded_torch" if variant == "cpu" else "embedded_torch_cuda"
    stdlib_file = "_stdlib_imports.txt" if variant == "cpu" else "_stdlib_imports_gpu.txt"

    # 1. 下载或确认模型
    model_src = ensure_model()

    # 2. 确保 dBert 权重就绪，并实跑注音链验证依赖完整
    ensure_dbert_artifacts()
    verify_annotator()

    # 3. 提取 embedded_torch 运行时
    subprocess.run([sys.executable, "prepare_embedded_torch.py", "--variant", variant], check=True)

    # 4. 收集标准库清单
    subprocess.run([sys.executable, "_collect_stdlib.py", "--target", emb_src, "--out", stdlib_file], check=True)

    # 5. 生成变体 spec
    spec_tmpl = (HERE / "FA-ASMR.spec").read_text(encoding="utf-8")
    spec_content = (spec_tmpl
                    .replace("@PROJECT_DIR@", str(HERE))
                    .replace("@EMBEDDED_TORCH@", emb_src)
                    .replace("@APP_NAME@", app_name))
    
    auto_spec = BUILD_ROOT / f"_auto_{variant}.spec"
    auto_spec.parent.mkdir(parents=True, exist_ok=True)
    auto_spec.write_text(spec_content, encoding="utf-8")

    # 5. 运行 PyInstaller
    stage_dist = BUILD_ROOT / f"_stage_{variant}"
    if stage_dist.exists():
        shutil.rmtree(stage_dist, ignore_errors=True)

    workpath = BUILD_ROOT / f"work_{variant}"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", str(stage_dist),
        "--workpath", str(workpath),
        str(auto_spec),
    ]
    log("运行 PyInstaller 打包 ...")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    log(f"PyInstaller 打包完成，耗时 {time.time()-t0:.1f}s")

    staged_app = stage_dist / app_name
    (staged_app / "_internal" / "variant.txt").write_text(variant + "\n", encoding="utf-8")

    # 6. 合并至 dist/<app_name>
    final_dir = DIST_ROOT / app_name
    final_dir.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(staged_app):
        rel = Path(root).relative_to(staged_app)
        dst_dir = final_dir / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_f = Path(root) / f
            dst_f = dst_dir / f
            shutil.copyfile(src_f, dst_f)

    # 7. 复制配置与 LoRA
    for cfg in ("fa_asmr_settings.json", "advanced_config.json"):
        if (HERE / cfg).is_file():
            shutil.copy2(HERE / cfg, final_dir / cfg)
    
    lora_dst = final_dir / "lora"
    lora_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "lora" / "fa_asmr_v9e10.pt", lora_dst / "fa_asmr_v9e10.pt")

    # 8. 写入启动脚本
    launcher_bat = final_dir / "运行FA-ASMR.bat"
    launcher_bat.write_text(f"""@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ============================================
echo   FA-ASMR V2 ({variant.upper()} 完整内置模型版)
echo ============================================
start "" "{app_name}.exe"
""", encoding="utf-8")

    # 9. 注入 1.26GB 模型
    model_dst_dir = final_dir / "models" / "hub" / "checkpoints"
    model_dst_dir.mkdir(parents=True, exist_ok=True)
    model_dst = model_dst_dir / "model.pt"
    log(f"复制 MMS-FA 模型权重 -> {model_dst} ...")
    shutil.copyfile(str(model_src), str(model_dst))

    # 10. 验证测试
    exe_path = final_dir / f"{app_name}.exe"
    log(f"验证可执行程序启动: {exe_path} ...")
    proc = subprocess.Popen([str(exe_path)], cwd=str(final_dir))
    time.sleep(4.0)
    ret = proc.poll()
    if ret is not None and ret != 0:
        fail(f"{app_name}.exe 启动异常退出，代码: {ret}")
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    log(f"[OK] {app_name}.exe 启动验证通过！")

    # 11. 压缩为发布资产
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pkg_name = f"FA-ASMR-v2.0.0-Windows-{variant.upper()}"
    log(f"正在进行 7-Zip 压缩: {pkg_name} ...")
    t0 = time.time()

    if variant == "cpu":
        # CPU 版单文件压缩
        out_zip = ARTIFACTS_DIR / f"{pkg_name}.zip"
        cmd = [SEVEN_ZIP, "a", "-tzip", "-mx=5", "-mmt=on", str(out_zip), app_name]
        subprocess.run(cmd, cwd=str(DIST_ROOT), check=True)
        log(f"CPU 单文件压缩完成: {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB, 耗时 {time.time()-t0:.1f}s)")
    else:
        # GPU 版分卷 1800m 压缩
        out_split = ARTIFACTS_DIR / f"{pkg_name}.zip"
        cmd = [SEVEN_ZIP, "a", "-tzip", "-mx=5", "-v1800m", "-mmt=on", str(out_split), app_name]
        subprocess.run(cmd, cwd=str(DIST_ROOT), check=True)
        parts = sorted(list(ARTIFACTS_DIR.glob(f"{pkg_name}.zip.*")))
        log(f"GPU 分卷压缩完成 ({len(parts)} 个分卷, 耗时 {time.time()-t0:.1f}s)")
        
        # 写入一键合并脚本
        merge_bat = ARTIFACTS_DIR / "merge_split_gpu_zip.bat"
        part_names = " + ".join([f'"{p.name}"' for p in parts])
        merge_bat.write_text(f"""@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo 正在合并分卷文件...
copy /b {part_names} "{pkg_name}.zip"
echo 合并完成: {pkg_name}.zip
pause
""", encoding="utf-8")

    log(f"[DONE] {variant.upper()} 构建与压缩全流程顺利完成！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["cpu", "gpu"], required=True)
    args = parser.parse_args()
    main(args.variant)

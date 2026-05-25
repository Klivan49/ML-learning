#!/usr/bin/env python3
"""
Build standalone executables with PyInstaller.

Usage:
  python scripts/build_exe.py              # build all
  python scripts/build_exe.py --gui-only   # only gui
  python scripts/build_exe.py --cli-only   # only cli sorting
"""

import os
import sys
import shutil
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")


def _pyinstaller(spec_name: str, script: str, console: bool, extra_hidden: list = None):
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--onefile",
        "--distpath", DIST,
        "--specpath", os.path.join(ROOT, "build"),
        "--workpath", os.path.join(ROOT, "build"),
        "--name", spec_name,
    ]
    if not console:
        cmd.append("--noconsole")
    cmd.append("--add-data")
    cmd.append(f"{os.path.join(ROOT, 'configs')}{os.pathsep}configs")
    cmd.append("--add-data")
    cmd.append(f"{os.path.join(ROOT, 'src')}{os.pathsep}src")
    cmd.append("--hidden-import")
    cmd.append("sklearn")
    cmd.append("--hidden-import")
    cmd.append("sklearn.ensemble")
    cmd.append("--hidden-import")
    cmd.append("sklearn.linear_model")
    cmd.append("--hidden-import")
    cmd.append("sklearn.preprocessing")
    cmd.append("--hidden-import")
    cmd.append("sklearn.pipeline")
    cmd.append("--hidden-import")
    cmd.append("joblib")
    cmd.append("--hidden-import")
    cmd.append("pandas")
    cmd.append("--hidden-import")
    cmd.append("numpy")
    cmd.append("--hidden-import")
    cmd.append("ttkbootstrap")
    if extra_hidden:
        for h in extra_hidden:
            cmd.extend(["--hidden-import", h])
    cmd.append(script)
    print(f"Building {spec_name}...")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Build executables")
    parser.add_argument("--gui-only", action="store_true")
    parser.add_argument("--cli-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    os.makedirs(DIST, exist_ok=True)

    gui_script = os.path.join(ROOT, "scripts", "gui.py")
    sort_script = os.path.join(ROOT, "scripts", "sort_files.py")
    train_script = os.path.join(ROOT, "scripts", "train_model.py")
    gen_script = os.path.join(ROOT, "scripts", "generate_dataset.py")

    if not args.cli_only:
        _pyinstaller("FileSorter", gui_script, console=False,
                     extra_hidden=["PIL", "PIL._tkinter_finder"])

    if not args.gui_only:
        _pyinstaller("sort-files", sort_script, console=True)
        _pyinstaller("train-model", train_script, console=True)
        _pyinstaller("generate-dataset", gen_script, console=True)

    # cleanup
    shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)

    print(f"\nDone! Executables in: {DIST}")
    for f in os.listdir(DIST):
        print(f"  {f}")


if __name__ == "__main__":
    main()

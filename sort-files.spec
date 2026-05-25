# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts/sort_files.py'],
    pathex=[],
    binaries=[],
    datas=[('/home/klivan/Проекты/Programming/ML-learning/configs', 'configs'), ('/home/klivan/Проекты/Programming/ML-learning/src', 'src')],
    hiddenimports=['sklearn', 'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.preprocessing', 'sklearn.pipeline', 'joblib', 'pandas', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='sort-files',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

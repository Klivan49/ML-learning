# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['src._venv_setup', 'configs.config', 'src.inference.predict', 'src.data_prep.dataset', 'src.models.train', 'src.models.model', 'sklearn', 'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.tree', 'sklearn.preprocessing', 'sklearn.pipeline', 'sklearn.metrics', 'sklearn.model_selection', 'sklearn.feature_extraction', 'pandas', 'numpy', 'joblib', 'ttkbootstrap']
tmp_ret = collect_all('ttkbootstrap')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\69mar\\Desktop\\УчЁБА\\ML-learning\\scripts\\gui.py'],
    pathex=['C:\\Users\\69mar\\Desktop\\УчЁБА\\ML-learning'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name='FileSorter',
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
    name='FileSorter',
)

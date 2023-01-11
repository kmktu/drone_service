# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['drone_main.py'],
    pathex=[],
    binaries=[],
    datas=[('SLowFast', 'SlowFast/'), ('Yolov5_StrongSORT_OSNet', 'Yolov5_StrongSORT_OSNet/'),
    ('detectron2-windows/detectron2', 'detectron2/'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/av', 'av'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/fvcore', 'fvcore'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/torch', 'torch'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/torchvision', 'torchvision'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/yacs', 'yacs'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/pytorchvideo', 'pytorchvideo'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/pycocotools', 'pycocotools'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/sklearn', 'sklearn'),
    ('C:/Users/mgkang.DI-SOLUTION/Anaconda3/envs/code-test/Lib/site-packages/joblib', 'joblib'),],
    hiddenimports=['PIL.ExifTags', 'logging.config', 'seaborn', 'easydict', 'gdown', 'lap', 'filterpy', 'av', 'fvcore',
    'torch', 'torchvision', 'detectron2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='drone_main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='drone_main',
)

echo off

python -m nuitka --standalone --show-progress --windows-disable-console --enable-plugin=pyside6 --plugin-enable=numpy --windows-icon-from-ico=icon.ico .\Batch3D.py

@REM mkdir .\main_ui.dist\trimesh
@REM cp -r C:\Users\tkf76\AppData\Roaming\Python\Python310\site-packages\trimesh\resources .\main_ui.dist\trimesh
echo off

python -m nuitka --standalone --show-progress --windows-disable-console --enable-plugin=pyside6 --plugin-enable=numpy --windows-icon-from-ico=icon.ico .\main_ui.py

mkdir .\main_ui.dist\trimesh
cp -r C:\Users\tkf76\AppData\Roaming\Python\Python310\site-packages\trimesh\resources .\main_ui.dist\trimesh
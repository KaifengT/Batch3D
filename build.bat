echo off

python -m nuitka --standalone --show-progress --windows-disable-console --enable-plugin=pyside6 --plugin-enable=numpy --windows-icon-from-ico=icon.ico .\Batch3D.py

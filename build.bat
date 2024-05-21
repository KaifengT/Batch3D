echo off

python -m nuitka --standalone --show-progress --windows-disable-console --enable-plugin=pyside6 --plugin-enable=numpy --windows-icon-from-ico=icon.ico .\main_ui.py
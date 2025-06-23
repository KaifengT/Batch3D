# build.py
import os
import subprocess
import sys

def compile_app():
    print(sys.executable)
    cmd = [
        sys.executable,
        "-m" ,
        "nuitka",
        "--standalone",
        "--show-progress",
        "--windows-console-mode=attach",

        "--nofollow-import-to=PySide6.QtPdf",
        "--nofollow-import-to=PySide6.QtNetwork",

        "--include-module=PySide6.QtWidgets",
        "--include-module=PySide6.QtCore",
        "--include-module=PySide6.QtGui",
        "--include-package=OpenGL_accelerate",

        "--follow-import-to=OpenGL_accelerate",

        "--enable-plugin=pyside6",

        "--windows-icon-from-ico=icon.ico",
        "Batch3D.py"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    compile_app()
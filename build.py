# build.py
import os
import subprocess
import sys
import shutil

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
        "--include-module=numpy.core",
        "--include-package=OpenGL_accelerate",

        "--follow-import-to=OpenGL_accelerate",

        "--enable-plugin=pyside6",

        "--windows-icon-from-ico=icon.ico",
        "Batch3D.py"
    ]
    return subprocess.run(cmd)
    

def post_build_copy():
    dist_dir = "Batch3D.dist"
    if not os.path.exists(dist_dir):
        print(f"Warning: Output directory '{dist_dir}' not found. Build might have skipped or failed.")
        return

    files_to_copy = [
        "ui",
        "example",
        "glw",
        "trimesh",
        "README.md",
        "icon.ico",
        "b3d.pyi"
    ]
    
    print(f"\n[Post-Build] Copying additional files to {dist_dir}...")
    for item in files_to_copy:
        src = os.path.abspath(item)
        dst = os.path.join(os.path.abspath(dist_dir), item)
        
        if not os.path.exists(src):
            print(f"  [Skip] Source not found: {item}")
            continue
            
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
                print(f"  [Dir ] Copied {item}")
            else:
                dst_folder = os.path.dirname(dst)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                shutil.copy2(src, dst)
                print(f"  [File] Copied {item}")
        except Exception as e:
            print(f"  [Error] Failed to copy {item}: {e}")
    print("[Post-Build] Finished.\n")

if __name__ == "__main__":
    compile_app()
    post_build_copy()
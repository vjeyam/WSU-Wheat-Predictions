from venv import create
import os
import sys
from subprocess import run, CalledProcessError

REQ_CANDIDATES = [
    os.path.join("installation", "requirements.txt"),
    "requirements.txt",
]

def find_requirements_file() -> str:
    for p in REQ_CANDIDATES:
        if os.path.exists(p):
            return p
    return ""

def venv_path() -> str:
    return os.path.join(os.getcwd(), "yolo8_venv")

def venv_python(venv_dir: str) -> str:
    # Windows vs. *nix
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")

def ensure_venv(venv_dir: str) -> None:
    if os.path.exists(venv_dir):
        print("[setup] Virtual env already exists:", venv_dir)
        return
    print("[setup] Creating virtual env:", venv_dir)
    create(venv_dir, with_pip=True)
    if not os.path.exists(venv_python(venv_dir)):
        raise RuntimeError("Virtual env creation failed (python not found).")

def pip(venv_dir: str, *args: str) -> None:
    py = venv_python(venv_dir)
    cmd = [py, "-m", "pip", *args]
    print("[setup] $", " ".join(cmd))
    r = run(cmd)
    if r.returncode != 0:
        raise CalledProcessError(r.returncode, cmd)

def main() -> int:
    req = find_requirements_file()
    if not req:
        print("ERROR: requirements.txt not found (looked for installation/requirements.txt and ./requirements.txt)")
        return 1
    print("[setup] Using requirements:", req)

    vdir = venv_path()
    ensure_venv(vdir)

    # make sure pip tooling is current
    try:
        pip(vdir, "install", "--upgrade", "pip", "setuptools", "wheel")
    except CalledProcessError:
        print("WARNING: failed to upgrade pip/setuptools/wheel, continuing...")

    # NOTE: remove subprocess32 (Python 2 only) if present
    # If you keep it in the file by accident, pip install will error on Py3.
    with open(req, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if "subprocess32" not in ln.strip().lower()]
    tmp = req + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(lines)

    try:
        pip(vdir, "install", "-r", tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print("\n[setup] Done.")
    print("[setup] Activate the environment then run your scripts:")
    if os.name == "nt":
        print("       .\\yolo8_venv\\Scripts\\activate")
    else:
        print("       source yolo8_venv/bin/activate")
    return 0

if __name__ == "__main__":
    sys.exit(main())
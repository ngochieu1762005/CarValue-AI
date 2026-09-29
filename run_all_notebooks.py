from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

if not notebooks:
    raise FileNotFoundError(f"No notebooks found in: {NOTEBOOK_DIR}")

# Make project modules importable inside executed notebooks on Windows/macOS/Linux.
env = os.environ.copy()
existing_pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

print(f"Python executable: {sys.executable}", flush=True)
print(f"Project root: {ROOT}", flush=True)
print(f"Found {len(notebooks)} notebooks.\n", flush=True)

for path in notebooks:
    print(f"=== Running {path.name} ===", flush=True)
    # Use the CURRENT Python interpreter instead of the `jupyter` executable.
    # This avoids Windows PATH errors such as WinError 2 when Jupyter is installed
    # but Python's Scripts directory is not on PATH.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=1200",
            "--ExecutePreprocessor.kernel_name=python3",
            str(path),
        ],
        cwd=str(ROOT),
        env=env,
        check=True,
    )
    print(f"PASS: {path.name}\n", flush=True)

print("All notebooks completed successfully.", flush=True)

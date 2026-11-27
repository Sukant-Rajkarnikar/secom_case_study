"""Reproduce everything in this folder from scratch.

Purpose: run steps 1-3 in order. Step 1 downloads SECOM from UCI; nothing else is needed.
Inputs:  none. Outputs: data/processed/ (not committed) and results/.
Run:     python run_pipeline.py   (about 5 minutes on a laptop)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["step1_data_split_folds.py", "step2_always_pass.py", "step3_team_pipeline.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / step)], check=True, cwd=ROOT)

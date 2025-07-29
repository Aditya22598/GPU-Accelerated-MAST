import json
import subprocess
from datetime import datetime

CELL_JSON = "/20TB-storage/aditya22598/zoo/cell_type_data.json"
RUN_SCRIPT = "run_celltype.py"

def run_for_celltype(ct):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Launching process for cell type: {ct}")
    try:
        result = subprocess.run(
            ["python", RUN_SCRIPT, ct],
            timeout=60*60*4  # 1 hour per cell type (adjust as needed)
        )
        if result.returncode != 0:
            print(f"!!! Error for {ct} (exit code {result.returncode}) !!!")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for cell type {ct}")

if __name__ == "__main__":
    with open(CELL_JSON) as f:
        cell_types = json.load(f)
    print(f"Starting pipeline for {len(cell_types)} cell types.")
    for ct in cell_types:
        run_for_celltype(ct)
    print("Pipeline completed.")


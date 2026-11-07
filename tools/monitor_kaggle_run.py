"""
Autonomous Kaggle Kernel Monitor and Artifact Downloader.
Polls status every 45s, streams log lines, and downloads output artifacts on completion.
"""

import os
import sys
import time
from pathlib import Path

import kaggle

KERNEL_SLUG = "ssachithananthan/p1-hvi-cidnet-training-t4-x2"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "weights" / "kaggle_run_v3"
LOG_FILE = REPO_ROOT / "outputs" / "kaggle_training_stream.log"

def main():
    api = kaggle.KaggleApi()
    api.authenticate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring Kaggle Kernel: {KERNEL_SLUG}...", flush=True)
    start_time = time.time()

    last_logged_lines = 0

    while True:
        status_res = api.kernels_status(KERNEL_SLUG)
        status_str = str(getattr(status_res, "status", status_res))
        fail_msg = getattr(status_res, "failure_message", None)
        elapsed_m = (time.time() - start_time) / 60.0

        print(f"[{elapsed_m:.1f}m] Status: {status_str}", flush=True)

        # Stream logs if available
        try:
            logs = api.kernels_logs(KERNEL_SLUG)
            if logs and isinstance(logs, str):
                lines = [l for l in logs.splitlines() if l.strip()]
                if len(lines) > last_logged_lines:
                    new_lines = lines[last_logged_lines:]
                    for l in new_lines[-5:]:
                        print(f"  > {l}", flush=True)
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        for l in new_lines:
                            f.write(l + "\n")
                    last_logged_lines = len(lines)
        except Exception:
            pass

        if "COMPLETE" in status_str.upper():
            print(f"\nKernel COMPLETED successfully in {elapsed_m:.1f} minutes!", flush=True)
            print(f"Downloading output artifacts to {OUTPUT_DIR}...", flush=True)
            api.kernels_output(KERNEL_SLUG, path=str(OUTPUT_DIR))
            print("Download completed!", flush=True)
            break
        elif any(err in status_str.upper() for err in ["ERROR", "FAILED", "CANCELLED"]):
            print(f"\nKernel terminated with state {status_str}: {fail_msg}", flush=True)
            sys.exit(1)

        time.sleep(45)

if __name__ == "__main__":
    main()

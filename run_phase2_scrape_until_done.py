#!/usr/bin/env python
"""
Run Phase 2 scrape-only in a loop until the exhibition file is fully generated.
On timeout, resume automatically (no --fresh). Then pause for review.
"""
import subprocess
import sys
from pathlib import Path

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"
TIMEOUT_SECONDS = 600  # 10 minutes per run
COMPLETE_MARKER = "Scrape complete (--scrape-only)"


def main():
    script_dir = Path(__file__).resolve().parent
    first = True
    while True:
        args = [sys.executable, "run_phase2_exhibitions.py", "--scrape-only"]
        if first:
            args.insert(-1, "--fresh")
            print("[run_phase2_scrape_until_done] First run: --fresh --scrape-only\n")
        else:
            print("[run_phase2_scrape_until_done] Resuming: --scrape-only (no --fresh)\n")
        try:
            result = subprocess.run(
                args,
                cwd=script_dir,
                timeout=TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            out = (result.stdout or "") + (result.stderr or "")
            print(out)
            if result.returncode == 0 and COMPLETE_MARKER in out:
                print("\n[run_phase2_scrape_until_done] Scrape fully generated. Pausing for review.")
                return 0
            print(f"\n[run_phase2_scrape_until_done] Incomplete (exit {result.returncode}) or timeout not reached. Resuming...")
        except subprocess.TimeoutExpired:
            print(f"\n[run_phase2_scrape_until_done] Timeout after {TIMEOUT_SECONDS}s. Resuming without --fresh...")
        first = False


if __name__ == "__main__":
    raise SystemExit(main() or 0)

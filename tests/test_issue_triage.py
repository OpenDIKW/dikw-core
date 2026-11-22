"""Include the standalone GitHub issue triage contract in the regular pytest gate."""

import subprocess
import sys
from pathlib import Path


def test_github_issue_triage() -> None:
    suite = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "issue_labels_test.py"
    subprocess.run([sys.executable, "-B", str(suite)], check=True)

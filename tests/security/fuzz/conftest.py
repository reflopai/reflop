# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

# The fuzzer reuses the corpus helpers and the parity seeds.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

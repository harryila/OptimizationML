#!/usr/bin/env python3
"""Theory-tree entry point for the canonical exact witness."""

from __future__ import annotations

import json

from passive_muon.witness import canonical_jordan_witness

if __name__ == "__main__":
    print(json.dumps(canonical_jordan_witness(), indent=2, sort_keys=True))

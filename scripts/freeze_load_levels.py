#!/usr/bin/env python
"""Write the calibrated load ladder into `orion.sim.load_levels`, or refuse.

The ladder is not hand-editable. `get_level` raises if CALIBRATED_LEVELS is empty,
so nothing in §Y can fire until this script has run, and this script writes the
table itself rather than trusting anyone to transcribe four floats correctly.

Three refusals, each for a failure that has actually happened here:

  * NOT MONOTONE in rho. If acceptance does not fall as offered load rises, the
    sweep did not measure what it claims and the levels are not a difficulty
    ladder.
  * DUPLICATE lambda. Two levels selecting the same sweep point is a degenerate
    ladder: the run completes, every cell carries a plausible number, and two
    columns of the results table are the same experiment. This happened once and
    was caught by eye.
  * ANCHOR NOT FOUND, i.e. the target file has drifted. The substitution is
    anchored on the assignment with `[^}]*`, NOT a `re.DOTALL` wildcard. A DOTALL
    wildcard once matched past the end of the dict and deleted the LoadLevel
    dataclass along with everything between it and the table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CALIB = Path("results/y3_load_calibration.json")
TARGET = Path("src/orion/sim/load_levels.py")
ANCHOR = re.compile(r"CALIBRATED_LEVELS: dict\[str, LoadLevel\] = \{[^}]*\}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", default=str(CALIB))
    ap.add_argument("--target", default=str(TARGET))
    ap.add_argument("--note", default="",
                    help="one line recorded in the header, e.g. what changed")
    args = ap.parse_args()

    res = json.loads(Path(args.calibration).read_text())

    if not res.get("monotone"):
        print("REFUSING to freeze: acceptance is not monotone in rho.", file=sys.stderr)
        return 2

    names = ["L1", "L2", "L3", "L4"]
    lams = [res["levels"][n]["lambda"] for n in names]
    if len({round(x, 6) for x in lams}) != len(lams):
        print("REFUSING to freeze: two levels share a lambda -> degenerate ladder.",
              file=sys.stderr)
        for n, x in zip(names, lams):
            print("    %s lambda=%.4f" % (n, x), file=sys.stderr)
        return 3

    rows = []
    for n in names:
        v = res["levels"][n]
        rows.append('    "%s": LoadLevel("%s", arrival_rate=%.4f, rho_offered=%.4f, '
                    'reference_acceptance=%.4f),'
                    % (n, n, v["lambda"], v["rho_offered"], v["acceptance_mean"]))

    header = (
        "#: FROZEN by scripts/freeze_load_levels.py from a §Y.3 calibration sweep:\n"
        "#:   N=%d, %d rho points, seeds %s, instances %s, %.0f s wall.\n"
        "#: Monotone in rho and every level has a distinct lambda; the script refuses\n"
        "#: to write this table otherwise, and writes it itself rather than by hand.\n"
        % (res["num_arrivals"], len(res["sweep"]), res["seeds"], res["instances"],
           res["wall_s"]))
    if args.note:
        header += "#: %s\n" % args.note

    block = header + "CALIBRATED_LEVELS: dict[str, LoadLevel] = {\n" + "\n".join(rows) + "\n}"

    p = Path(args.target)
    s = p.read_text()
    s2, n = ANCHOR.subn(block, s, count=1)
    if n != 1:
        print("REFUSING to freeze: anchor not found in %s (has it drifted?)" % p,
              file=sys.stderr)
        return 4
    p.write_text(s2)

    print("FROZEN at N=%d:" % res["num_arrivals"])
    for r in rows:
        print("   ", r.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

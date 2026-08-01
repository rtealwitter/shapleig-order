# Generate the fixed paired-extremes schedules for the player counts used in
# the replication (p = 9, 14, 16; p = 10 already exists). The schedule is the
# frozen-symmetric-kernel greedy from gp_core.fixed_schedule: complement-paired
# extremes first, then balanced middles. Length covers every iteration of the
# corresponding sweep plus slack for init-design collisions. Resumable: skips
# schedules whose file already exists.
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gp_core import fixed_schedule

OUT = os.path.join(HERE, "..", "shapleig-repo", "data")
TARGETS = {9: 512, 14: 600, 16: 600}  # p -> schedule length

for p, m_max in TARGETS.items():
    path = os.path.join(OUT, f"paired_schedule_p{p}.npy")
    if os.path.exists(path):
        print(f"skip p={p} (exists)")
        continue
    t0 = time.time()
    sched = fixed_schedule(p, m_max)
    np.save(path, sched)
    sizes = [bin(int(m)).count("1") for m in sched[: 2 * p + 6]]
    print(f"p={p}: {m_max} steps in {time.time() - t0:.0f}s; "
          f"head sizes {sizes}")

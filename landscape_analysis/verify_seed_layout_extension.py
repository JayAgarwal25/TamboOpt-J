import glob
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
OTHER_OPT_DIR = os.path.join(BASE, "other_optimizers")

seed_dirs = sorted(
    d for d in glob.glob(os.path.join(OTHER_OPT_DIR, "seed*"))
    if os.path.isdir(d)
)

print(f"Found {len(seed_dirs)} seed layout directories")
print()

print("=== detector removal: U at 20-detector floor ===")
print(f"{'layout':<16}{'fullU':>9}{'redund@20':>11}{'redund%':>9}{'crit@20':>10}{'crit%':>8}{'rand@20':>9}{'rand%':>8}")
for d in seed_dirs:
    tag = os.path.basename(d)
    text = open(os.path.join(d, "detector_removal_analysis_out.txt")).read()
    full_u = float(re.search(r"Full 100-detector U:\s*([\d.]+)", text).group(1))
    crit = float(re.search(r"highest_dip_first\s*:\s*U at 20 remaining\s*=\s*([\d.]+)", text).group(1))
    redund = float(re.search(r"lowest_dip_first\s*:\s*U at 20 remaining\s*=\s*([\d.]+)", text).group(1))
    rand = float(re.search(r"random\s*:\s*U at 20 remaining\s*=\s*([\d.]+)", text).group(1))
    print(
        f"{tag:<16}{full_u:>9.2f}{redund:>11.2f}{100*redund/full_u:>8.1f}%"
        f"{crit:>10.2f}{100*crit/full_u:>7.1f}%{rand:>9.2f}{100*rand/full_u:>7.1f}%"
    )

print()
print("=== leave-one-out dip stats ===")
print(f"{'layout':<16}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
for d in seed_dirs:
    tag = os.path.basename(d)
    text = open(os.path.join(d, "detector_removal_analysis_out.txt")).read()
    m = re.search(r"dip stats:\s*min=([\-\d.]+)\s*max=([\-\d.]+)\s*mean=([\-\d.]+)\s*std=([\-\d.]+)", text)
    mn, mx, mean, std = (float(x) for x in m.groups())
    print(f"{tag:<16}{mn:>9.3f}{mx:>9.3f}{mean:>9.3f}{std:>9.3f}")

print()
print("=== full-space 2D slice spans (% of optimized base U) ===")
print(f"{'layout':<16}{'pair0%':>9}{'pair1%':>9}{'argmax_gain_pair0':>19}{'argmax_gain_pair1':>19}")
for d in seed_dirs:
    tag = os.path.basename(d)
    text = open(os.path.join(d, "full_space_2d_slice_out.txt")).read()
    m0 = re.search(r"optimized_pair0\s*:.*?\(([\d.]+)% of base U\)\s*gain=([+\-\d.]+)", text)
    m1 = re.search(r"optimized_pair1\s*:.*?\(([\d.]+)% of base U\)\s*gain=([+\-\d.]+)", text)
    p0, g0 = float(m0.group(1)), float(m0.group(2))
    p1, g1 = float(m1.group(1)), float(m1.group(2))
    print(f"{tag:<16}{p0:>8.2f}%{p1:>8.2f}%{g0:>19.3f}{g1:>19.3f}")

print()
print("=== detector grid scan: center/edge gains ===")
print(f"{'layout':<16}{'center_gain':>13}{'edge_gain':>11}")
for d in seed_dirs:
    tag = os.path.basename(d)
    text = open(os.path.join(d, "detector_grid_scan_out.txt")).read()
    mc = re.search(r"center\s*\(idx=\s*\d+\):.*?gain=([+\-\d.]+)", text)
    me = re.search(r"edge\s*\(idx=\s*\d+\):.*?gain=([+\-\d.]+)", text)
    gc, ge = float(mc.group(1)), float(me.group(1))
    print(f"{tag:<16}{gc:>13.3f}{ge:>11.3f}")

print()
print("=== error log check ===")
n_nonempty = 0
for d in seed_dirs:
    for errf in glob.glob(os.path.join(d, "*_err.txt")):
        if os.path.getsize(errf) > 0:
            n_nonempty += 1
            print(f"NON-EMPTY: {errf}")
print(f"{n_nonempty} non-empty error logs out of 30 jobs (3 scripts x 10 layouts)")

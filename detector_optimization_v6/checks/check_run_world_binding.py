"""Guard: no script may bind a run-world path at import time.

This is the structural half of the run_world fix. The refactor removed every
import-time binding of the `constants.py` path globals, but nothing stopped the
next script from reintroducing one, and the failure is silent by construction:
the script runs, produces plausible numbers, and attributes them to the wrong
run. Four results shipped that way before the pattern was noticed.

So the rule is checked rather than remembered. This walks every .py under the v6
tree and fails on:

  1. importing a path constant from `constants` (binds at import, before argv)
  2. reaching through the module (`_C.FNN_FOLDER`) to the same effect
  3. calling `run_world.resolve(args)` without `add_run_world_args(ap)`, which
     always exits since the flags it reads were never registered
  4. calling `load_models` without naming both folders

Run it directly, or from a pre-commit hook:

    python checks/check_run_world_binding.py
"""
import ast
import os
import sys

PATH_CONSTANTS = {
    "RUN_LOCATION", "SHOWER_CACHE", "TRAINING_DATASET_FOLDER", "FNN_FOLDER",
    "RECON_FOLDER", "OPT_FOLDER", "DUAL_SHOWER_CACHE_PATH", "TAU_CORPUS_PATH",
    "DUAL_SPECIES_IDS_PATH", "DUAL_POSITIONS_PATH", "HELDOUT_SHOWER_CACHE_PATH",
    "HELDOUT_SPECIES_IDS_PATH", "HELDOUT_POSITIONS_PATH",
}
EXEMPT = {"modules_v6/run_world.py", "modules_v6/constants.py",
          "checks/check_run_world_binding.py"}


def iter_sources(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if rel.replace(os.sep, "/") in EXEMPT:
                continue
            yield rel, path


def check_file(rel, path):
    src = open(path).read()
    tree = ast.parse(src, filename=path)
    problems = []
    resolves_with_args = []
    registers = False

    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module
                and node.module.endswith("constants")):
            hits = sorted({a.name for a in node.names} & PATH_CONSTANTS)
            if hits:
                problems.append((node.lineno,
                                 f"imports {', '.join(hits)} from constants; these bind "
                                 f"at import, before argv exists. Use "
                                 f"run_world.resolve(args) instead."))
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in ("_C", "constants")
                and node.attr in PATH_CONSTANTS):
            problems.append((node.lineno,
                             f"reaches through to {node.value.id}.{node.attr}; same "
                             f"import-time binding by another route."))
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if (name == "resolve" and isinstance(fn, ast.Attribute)
                    and getattr(fn.value, "id", "") == "run_world"
                    and node.args):
                resolves_with_args.append(node.lineno)
            if name == "add_run_world_args":
                registers = True
            if name == "load_models":
                kw = {k.arg for k in node.keywords}
                if not {"fnn_folder", "recon_dir"} <= kw:
                    problems.append((node.lineno,
                                     "load_models without both fnn_folder= and "
                                     "recon_dir=; it would fall back to whichever "
                                     "checkpoints the constants happen to name."))

    if resolves_with_args and not registers:
        problems.append((resolves_with_args[0],
                         "calls run_world.resolve(args) but never registers the "
                         "flags with add_run_world_args(ap), so it always exits."))
    return problems


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    failures = 0
    scanned = 0
    for rel, path in iter_sources(root):
        scanned += 1
        for lineno, msg in check_file(rel, path):
            where = f"{rel}:{lineno}" if lineno else rel
            print(f"FAIL {where}\n     {msg}")
            failures += 1
    print(f"\nscanned {scanned} files, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

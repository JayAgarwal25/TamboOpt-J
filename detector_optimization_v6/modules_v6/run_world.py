"""Explicit run-world resolution: which folders a script reads and writes.

The problem this exists to solve. Every path in `constants.py` is a module-level
string, so "which run am I operating on" is decided at IMPORT time, before any
script has looked at its own command line. A script launched without the right
override flag therefore does not fail — it silently binds to whatever run world
`constants.py` happened to point at when the module was imported, runs to
completion, and prints confident numbers about the wrong artifacts. Four
separate results shipped that way: a stage-4 run that tried to write into
another user's world, and three evaluators that scored a stale surrogate.

The fix is to make the binding late and loud:

  * `resolve()` picks the world AFTER argv exists, from an explicit source.
  * Falling back to `constants.py` is an ERROR unless the caller opts in with
    `--use-constants-run-world`, so "I forgot the flag" stops the job in the
    first second instead of producing a plausible wrong answer hours later.
  * Whatever is chosen is PRINTED, with the precedence level that supplied it,
    so the provenance is in the log of every run rather than inferred later.
  * `need_write=True` checks writability up front, turning the stage-4
    PermissionError into a one-line message before any compute is spent.

Usage, script with argparse:

    ap = argparse.ArgumentParser()
    run_world.add_run_world_args(ap)
    args = ap.parse_args()
    W = run_world.resolve(args, need_write=True)
    primary = torch.load(os.path.join(W.dataset_folder, "primary.pt"))

Usage, top-level script with no argparse (it parses argv itself, tolerantly):

    W = run_world.resolve()
"""

import argparse
import json
import os
import pwd
import sys
import time
from dataclasses import dataclass, asdict, fields

from modules_v6 import constants as _C


ENV_VAR          = "TAMBO_RUN_WORLD"
MANIFEST_NAME    = "run_world.json"
STAMP_NAME       = "_run_world_stamp.json"

# Subfolder basenames used when a --run_world root carries no manifest. Derived
# from constants so a bare root reproduces the historical layout, but a world
# that renames a stage (the 02/03 "_recentered" suffix, say) must ship a
# manifest — a bare root cannot express it and resolve() says so.
_DEFAULT_SUBDIRS = {
    "shower_cache":   os.path.basename(_C.SHOWER_CACHE),
    "dataset_folder": os.path.basename(_C.TRAINING_DATASET_FOLDER),
    "fnn_folder":     os.path.basename(_C.FNN_FOLDER),
    "recon_folder":   os.path.basename(_C.RECON_FOLDER),
    "opt_folder":     os.path.basename(_C.OPT_FOLDER),
}

_PATH_FIELDS = ("shower_cache", "dataset_folder", "fnn_folder",
                "recon_folder", "opt_folder")

# Two of the five are PREFIXES, not folders: each stage appends its own tag
# ("_deepsets" for recon, "_lbfgs_ensemble_full_corpus_{scheme}" for stage 4),
# so the bare path legitimately does not exist on disk. Flagging them as
# missing on every run would train the reader to ignore the banner, which is
# exactly the habit this module exists to break.
_PREFIX_FIELDS = ("recon_folder", "opt_folder")


def _exists_mark(path: str, is_prefix: bool) -> str:
    if os.path.isdir(path):
        return ""
    if is_prefix:
        parent, base = os.path.dirname(path), os.path.basename(path)
        try:
            hits = sorted(d for d in os.listdir(parent)
                          if d.startswith(base) and os.path.isdir(os.path.join(parent, d)))
        except OSError:
            hits = []
        if hits:
            shown = ", ".join(hits[:3]) + (f", +{len(hits) - 3} more" if len(hits) > 3 else "")
            return f"   [prefix -> {shown}]"
    return "   [MISSING]"


@dataclass(frozen=True)
class RunWorld:
    """One run's folder set, resolved from an explicit source.

    `source` records WHICH precedence level supplied the root (flag, env,
    manifest, constants) and is printed and stamped, so a result artifact can
    always be traced back to how its inputs were chosen.
    """
    root: str
    shower_cache: str
    dataset_folder: str
    fnn_folder: str
    recon_folder: str
    opt_folder: str
    source: str

    # The two stage folders whose on-disk names carry a suffix the stage itself
    # appends. Kept as properties so callers stop rebuilding the string.
    @property
    def recon_dir(self) -> str:
        """03_train_recon_deepsets.py's actual output folder."""
        return self.recon_folder + "_deepsets"

    def opt_dir(self, suffix: str) -> str:
        """A stage-4 output folder; `suffix` is the optimizer/scheme tag."""
        return self.opt_folder + suffix

    def describe(self) -> str:
        lines = [f"run world   : {self.root}",
                 f"  source    : {self.source}"]
        for f in _PATH_FIELDS:
            p = getattr(self, f)
            lines.append(f"  {f:<14}: {p}{_exists_mark(p, f in _PREFIX_FIELDS)}")
        return "\n".join(lines)

    def stamp(self, folder: str, **extra) -> None:
        """Record into `folder` which world and inputs produced its contents."""
        os.makedirs(folder, exist_ok=True)
        rec = {"run_world": asdict(self),
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "argv": sys.argv, **extra}
        tmp = os.path.join(folder, STAMP_NAME + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(tmp, os.path.join(folder, STAMP_NAME))


def add_run_world_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --run_world, its per-folder escapes, and the constants opt-in."""
    g = ap.add_argument_group("run world")
    g.add_argument("--run_world", type=str, default=None,
                   help=f"Root folder of the run to operate on. Reads "
                        f"<root>/{MANIFEST_NAME} for its stage folder names if "
                        f"present, else assumes the default names. Overrides "
                        f"${ENV_VAR}.")
    g.add_argument("--use-constants-run-world", action="store_true",
                   help="Permit falling back to the paths hardcoded in "
                        "constants.py. Without this, a run with no --run_world "
                        "and no $" + ENV_VAR + " is an error rather than a "
                        "silent bind to whatever constants.py points at.")
    for f in _PATH_FIELDS:
        g.add_argument(f"--{f}", type=str, default=None,
                       help=f"Override just this folder, after --run_world.")
    return ap


def _tolerant_parse(argv=None):
    """Parse only the run-world flags, ignoring everything else.

    For the top-level scripts that have no argparse of their own: they can still
    be pointed at a world without acquiring a full CLI.
    """
    ap = argparse.ArgumentParser(add_help=False)
    add_run_world_args(ap)
    known, _ = ap.parse_known_args(argv)
    return known


def _manifest_subdirs(root: str):
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        man = json.load(fh)
    missing = [f for f in _PATH_FIELDS if f not in man]
    if missing:
        sys.exit(f"[run_world] {path} is missing required keys: {missing}")
    return {f: man[f] for f in _PATH_FIELDS}


def _owner(path: str) -> str:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return "?"
        probe = parent
    try:
        return pwd.getpwuid(os.stat(probe).st_uid).pw_name
    except (KeyError, OSError):
        return "?"


def _check_writable(world: "RunWorld") -> None:
    """Fail before any compute if an output folder cannot be written.

    Checks the nearest EXISTING ancestor, because stage folders are normally
    created by the stage itself; the question is whether creating them will be
    permitted, not whether they are there yet.
    """
    me = pwd.getpwuid(os.getuid()).pw_name
    bad = []
    for f in _PATH_FIELDS:
        target = getattr(world, f)
        probe = target
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not os.access(probe, os.W_OK):
            bad.append((f, target, probe, _owner(probe)))
    if bad:
        msg = [f"[run_world] not writable as user '{me}':"]
        for f, target, probe, own in bad:
            msg.append(f"  {f}: {target}")
            msg.append(f"      blocked at {probe}  (owner: {own})")
        msg.append("Point --run_world at a world you own, or drop need_write "
                   "if this script only reads.")
        sys.exit("\n".join(msg))


def resolve(args=None, *, need_write: bool = False, argv=None,
            quiet: bool = False) -> RunWorld:
    """Resolve the run world from an explicit source, or exit.

    Precedence, highest first:
      1. per-folder flags (--fnn_folder etc), applied on top of whatever 2-4 give
      2. --run_world <root>
      3. $TAMBO_RUN_WORLD
      4. constants.py, ONLY with --use-constants-run-world
    """
    if args is None:
        args = _tolerant_parse(argv)

    root_flag = getattr(args, "run_world", None)
    env_root  = os.environ.get(ENV_VAR) or None
    allow_c   = getattr(args, "use_constants_run_world", False)

    if root_flag:
        root, source = os.path.abspath(root_flag), "--run_world"
    elif env_root:
        root, source = os.path.abspath(env_root), f"${ENV_VAR}"
    elif allow_c:
        root, source = _C.RUN_LOCATION, "constants.py (--use-constants-run-world)"
    else:
        sys.exit(
            "[run_world] no run world specified.\n"
            "  This script will NOT silently fall back to constants.py, because\n"
            "  that is how four separate runs shipped results computed against\n"
            "  the wrong artifacts.\n"
            "  Choose one:\n"
            f"    --run_world <root>              operate on that run\n"
            f"    export {ENV_VAR}=<root>   same, for a whole batch script\n"
            f"    --use-constants-run-world       accept the constants.py default:\n"
            f"        {_C.RUN_LOCATION}")

    subdirs = None
    if source != "constants.py (--use-constants-run-world)":
        subdirs = _manifest_subdirs(root)
        if subdirs is not None:
            source += f" + {MANIFEST_NAME}"
        else:
            subdirs = dict(_DEFAULT_SUBDIRS)
            source += " (no manifest, default stage names)"
        paths = {f: os.path.join(root, subdirs[f]) for f in _PATH_FIELDS}
    else:
        paths = {"shower_cache":   _C.SHOWER_CACHE,
                 "dataset_folder": _C.TRAINING_DATASET_FOLDER,
                 "fnn_folder":     _C.FNN_FOLDER,
                 "recon_folder":   _C.RECON_FOLDER,
                 "opt_folder":     _C.OPT_FOLDER}

    overridden = []
    for f in _PATH_FIELDS:
        v = getattr(args, f, None)
        if v:
            paths[f] = os.path.abspath(v)
            overridden.append(f)
    if overridden:
        source += f" + per-folder flags {overridden}"

    world = RunWorld(root=root, source=source, **paths)

    if need_write:
        _check_writable(world)
    if not quiet:
        print(world.describe(), flush=True)
    return world


def describe_file(path: str) -> str:
    """Absolute path + mtime + size of a checkpoint, for load-time logging.

    Three evaluators scored a stale surrogate and shipped the numbers because
    the load line named only the FILE ("fnn_electron.pt"), which is identical in
    every run world. The folder and the mtime are what distinguish an August 1
    checkpoint from an August 12 one, so they belong in the log line.
    """
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
    except OSError as e:
        return f"{ap}  [UNREADABLE: {e.strerror}]"
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
    return f"{ap}  mtime={mtime}  {st.st_size / 1e6:.1f} MB"

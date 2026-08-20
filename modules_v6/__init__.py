"""v6 modules package.

Self-contained: `modules_v6.legacy_core` vendors the geometry, response-kernel
and utility code that earlier generations loaded from separate version folders
(now retired to the `legacy-full-repo` branch). Importing this package has no
side effects — it does not touch `sys.path`.

One external dependency remains, deliberately un-vendored:
`legacy_core.generate_showers` needs the sibling repo at `TAMBO-opt`
(`util.allshowers_related.generate_showers`). See that file's docstring.
"""

# `interpretability/` — what the trained models actually use

Probing, not evaluation. `eval/` asks how good an output is; this asks which
information a representation carries, by fitting a readout from that
representation to a target and reporting held-out R^2.

## Files

| file | role |
|---|---|
| `probes.py` | shared readout probes: linear (closed-form ridge) and MLP, a seeded split, and R^2 |
| `surface_probe.py` | whether a candidate input feature is already implied by the coordinates |

## How to read a probe

A single R^2 means very little. The reading always comes from a CONTRAST:

- linear vs MLP on the same features separates "linearly decodable" from
  "decodable at all";
- a rich feature block vs a trivial one (raw coordinates, say) says whether the
  rich block adds anything.

If the trivial block already scores near 1, the comparison is inconclusive, not
negative: there was no headroom for the richer block to demonstrate anything.
Report it that way.

Probes are cheap. Running one before committing to a training run has already
ruled out a proposed input feature that would otherwise have cost a retrain.

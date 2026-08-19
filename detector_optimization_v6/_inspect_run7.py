import json
d = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar/run 7 6 chains on top of run 6/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_center"
log = json.load(open(f"{d}/optimize_log.json"))
print(type(log))
if isinstance(log, dict):
    print(list(log.keys()))
    for k, v in log.items():
        print(k, type(v), (len(v) if hasattr(v, '__len__') else v))
elif isinstance(log, list):
    print(len(log))
    print(log[0])
    print(log[-1])

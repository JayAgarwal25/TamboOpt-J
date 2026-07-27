evaluation based on "detector_optimization_v6/plots/eval_true_utility.py"

on one batch 512

CENTER LAYOUT (baseline) vs OPTIMIZED LAYOUT
                  surrogate-U     true-U
  baseline center       13.5307        8.8140
  optimized          170.2994       36.2635

  ΔU surrogate (opt - center) : +156.7687
  ΔU true      (opt - center) : +27.4496
  artifact gap (surr - true, optimized) : +134.0359

  VERDICT: PARTIAL — some real gain, but the surrogate overstates it.

  component breakdown (surrogate | true), optimized layout:
    u_theta    +85.4485 |  +20.3930
    u_phi      +62.4745 |   +9.8946
    u_e        +22.3764 |   +5.9759
    u_pr      +11313.7090 | +11313.7090


GRID_LAYOUT (baseline) vs OPTIMIZED LAYOUT
                  surrogate-U     true-U
  baseline grid      169.8399       35.8354
  optimized          168.9291       36.9783

  ΔU surrogate (opt - grid) : -0.9108
  ΔU true      (opt - grid) : +1.1430
  artifact gap (surr - true, optimized) : +131.9508

  VERDICT: optimizer did not raise even surrogate-U here (check the run).

  component breakdown (surrogate | true), optimized layout:
    u_theta    +84.2130 |  +21.4377
    u_phi      +62.6281 |   +9.5921
    u_e        +22.0880 |   +5.9485
    u_pr      +11313.7090 | +11313.7090

on full database

GRID LAYOUT (baseline) vs OPTIMIZED LAYOUT
                  surrogate-U     true-U
  baseline grid      169.8399       35.8354
  optimized          176.7188       34.4832

  ΔU surrogate (opt - grid) : +6.8789
  ΔU true      (opt - grid) : -1.3522
  artifact gap (surr - true, optimized) : +142.2356

  VERDICT: ARTIFACT — surrogate-U rose but true-U did not; the movement exploits the surrogate, not the physics.

  component breakdown (surrogate | true), optimized layout:
    u_theta    +86.1348 |  +19.7701
    u_phi      +67.9285 |   +8.8084
    u_e        +22.6556 |   +5.9046
    u_pr      +11313.7090 | +11313.7090



validation redone in "detector_optimization_v6/notebooks/recon_grid_score_diagnostics_deepsets.ipynb"

limitations found in the training in "detector_optimization_v6/04_optimize_lbfgs_ensemble.py", where a single small batch was used by lbfgs for fine-tuning

the code for lbfgs is updated so it should perform better now

several runs:
    - test_v6_run_04_optimize_lbfgs_ensemble... - run with disabled seed for batch (512) selection in "04_optimize_lbfgs_ensemble", but still the same one batch (512) for all 15 optimizations
    - test_v6_run_04_optimize_lbfgs_ensemble..._seed_42 - run with 42 seed for batch (512) selection in "04_optimize_lbfgs_ensemble"
    - test_v6_run_04_optimize_lbfgs_ensemble_full_corpus... - runn batches of 15k and optimize on the whole dataset, last code update

TODO 1: biggest issue at the moment is the artifact gap - difference between surrogate and training data for utility calculation
TODO 2: can all the data from the dataset be used efficiently in lbfgs
TODO 3: in yhe full lbfgs run the grid was optimized on the surrogate, but the raw showers have worse reconstruction performance, how to fix that. maybe depends on TODO 1


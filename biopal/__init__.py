"""BiOPAL: Bilevel OPtimizer with an Agent in the Loop 
This instance: a biological expert as the outer level of bilevel fine-tuning; OPAL stays the inner level.

Everything runs through finetune/finetune.py (--mode biopal, or --mode keep for the offline twin):
  panel.py    data roles (fit F, diagnostic D, private A), cohorts, names shown to the expert
  measure.py  checkpoint measurements: signed loss gaps per cohort and layer
  outer.py    gates: measure, ask, matched trial on A, adopt or keep
  expert.py   the expert call (Claude CLI, a local open-weights model, or scripted) and its checks
  replay.py   python -m biopal.replay: other proposers at a run's saved gates, joined to the recorded trials
"""

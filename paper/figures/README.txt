Figures directory — The Agentic Chemometrician paper
=====================================================

Figures to be placed here before camera-ready submission:

fig1_architecture.pdf
  System architecture diagram.
  Shows the three-layer stack: tools/ (MCP server, thin JSON wrappers) ->
  core/ (deterministic, no MCP imports) -> contracts layer (frozen dataclasses).
  Also shows the HITL gate positions (propose_analysis_plan, select_best_model,
  recommend_next_model) and the bidirectional data flow between layers.
  Data source: manual diagram (draw.io or TikZ), based on src/ codebase structure.

fig2_cv_gap.pdf
  Naive KFold vs. GroupKFold / LeaveOneGroupOut cross-validation accuracy gap.
  Bar chart or box plot: per-scenario mean accuracy under naive CV vs. grouped CV.
  Data source: Layer-1 evaluation harness output (run_scenario_full, seed=42).
  TODO: generate from harness run.

fig3_leakage_roc.pdf  (optional)
  ROC curve or precision-recall curve for the replicate/group leakage detector
  across the 6 fixtures (3 positive, 3 clean) in tests/fixtures/eval/.
  Data source: compute_leakage_detection() in src/chemometrics_mcp/core/evaluation.py.
  TODO: generate from harness run.

Additional figures (if space permits):
  fig4_scenario_metrics.pdf — per-scenario accuracy/RMSE bar chart.
  fig5_ablation.pdf        — ablation delta heatmap (Table 5 visualised).

# Legacy/preliminary results notice

All existing Phase 6 notebooks, checkpoints, predictions, figures, configuration summaries, and metrics in this directory are historical/preliminary artifacts. They have been preserved unchanged.

Stage 0/1A review found that the historical auxiliary labels are deterministic proxies of the main three-class label, the lightweight hate head is wired to the sarcasm target, the lightweight Bangla class weights are unused, and the directories called “separate encoder” contain single-encoder head ablations. There is no implemented GRL/domain classifier.

Correcting these issues creates new experiments and new metrics. Corrected runs must use the isolated `corrected_pipeline`, write only to new `corrected_*` output directories, and must never overwrite the files here or the historical master-results table.

See `../EXPERIMENT_AUDIT.md`, `../LABEL_RECONCILIATION.md`, and `../STAGE1A_REPORT.md` for details.

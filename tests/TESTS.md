Test notes

Scope
- Focus on core behavior in feedback collection and monitoring.
- Keep tests small and local, using temp paths where files are written.

tests/unit/monitoring/test_feedback_collector.py
- test_save_and_load_feedback_roundtrip: ensure a feedback record is saved, then loaded with input features expanded into columns.
- test_save_feedback_appends_records: ensure multiple records append to the same CSV file.

tests/unit/monitoring/test_feedback_monitor.py
- test_compute_psi_zero_for_identical_distributions: PSI should be near zero for identical numeric distributions.
- test_compute_psi_returns_none_for_constant_series: PSI is not computed when there is no variation.
- test_evaluate_feedback_metrics: verify basic feedback metrics and derived range stats.

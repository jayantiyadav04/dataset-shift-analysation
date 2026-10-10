# Development fixtures

`shiftlab.fixtures.synthetic_training_view()` creates a deterministic, tiny offline
fixture in memory. It supplies source labels and target adaptation features only.
`shiftlab doctor` uses it to verify the installed estimator APIs with sample weights.
No accuracy, macro F1, adaptation benefit or real dataset findings are reported.

Optional development outputs belong in `examples/generated/`, which is ignored by
Git. They must never be registered as a completed study in `artifacts/`.

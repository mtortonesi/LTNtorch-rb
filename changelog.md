# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LTN::FuzzyOps`, a port of LTNtorch's `ltn.fuzzy_ops`: the projections
  `pi_0`/`pi_1`; the connective operators (`NotStandard`, `NotGodel`,
  `AndMin`, `AndProd`, `AndLuk`, `OrMax`, `OrProbSum`, `OrLuk`,
  `ImpliesKleeneDienes`, `ImpliesGodel`, `ImpliesReichenbach`,
  `ImpliesGoguen`, `ImpliesLuk`, `Equiv`); the aggregators (`AggregMin`,
  `AggregMean`, `AggregPMean`, `AggregPMeanError`), with masks for guarded
  quantification; and `SatAgg`. Optional arguments are keywords
  (`AndProd.new(stable: false)`, `agg.call(xs, dim: [0], p: 3)`), and
  Python's ValueError becomes ArgumentError.
- `LTN::LTNObject`, a grounding together with the labels of its free
  variables.
- Reference values for every operator and aggregator, gradients included,
  generated from LTNtorch by `test/fixtures/fuzzy_ops.py`.
- Project scaffolding: `ltntorch` gem with the `LTN` module, `LTN.device`
  configuration, sus/covered test setup, standard linting, and CI on
  Ubuntu (every push) and macOS (scheduled).

### Fixed (compared to LTNtorch)

- `SatAgg` reports the shapes of formulas that are not closed; LTNtorch
  raises an unrelated error while building that message.

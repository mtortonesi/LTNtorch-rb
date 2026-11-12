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
- The core of LTN, ported from LTNtorch's `ltn.core`: `LTN::Constant`
  (optionally trainable), `LTN::Variable`, `LTN.process_ltn_objects` (LTN
  broadcasting), `LTN::Predicate` and `LTN::Function` (grounded by a
  `Torch::NN::Module`, a function given as `func:`, or a block),
  `LTN::LambdaModel`, `LTN.diag`/`LTN.undiag`, `LTN::Connective` and
  `LTN::Quantifier` (with guarded quantification, the guard given as
  `cond_fn:` or a block). Quantifiers take `:forall`/`:exists` (LTNtorch's
  `"f"`/`"e"` work too).
- Ruby ports of LTNtorch's seven examples (`examples/`, with the Iris,
  crabs and real estate datasets; MNIST is downloaded on first use) and four
  tutorials (`tutorials/`), smoke-tested by `test/examples.rb`, plus a
  seeded learning benchmark of the examples (`benchmark/examples.rb`), run
  by CI as a separate job.
- Example 4 takes `--default-init` (`init_weights: false`) to keep PyTorch's
  default weight initialization: about 98% test accuracy on MNIST addition,
  as in the LTN paper, instead of the about 48% of LTNtorch's notebook
  initialization, which stays the default.
- `test/ltn/training_parity.rb`: training a small knowledge base from the
  same data and weights follows LTNtorch's loss at every epoch, from
  `test/fixtures/training.py`.
- Reference results for the core (broadcasting, predicates, functions,
  connectives, quantifiers, guarded and diagonal quantification, and the
  gradients of a small knowledge base), generated from LTNtorch by
  `test/fixtures/core.py`; float64 and scalar cases for the fuzzy operators.
- Reference values for every operator and aggregator, gradients included,
  generated from LTNtorch by `test/fixtures/fuzzy_ops.py`.
- Project scaffolding: `ltntorch` gem with the `LTN` module, `LTN.device`
  configuration, sus/covered test setup, standard linting, and CI on
  Ubuntu (every push) and macOS (scheduled).

### Changed (compared to LTNtorch)

- `LTN.process_ltn_objects`, `Quantifier.compute_mask` and
  `Quantifier.transpose_vars` never modify their arguments. LTNtorch rebinds
  the value and free variables of intermediate results it processes, so
  reusing a formula after, e.g., a connective sees the broadcast grounding;
  the computed values are the same.
- `LTN::Connective#call` passes keyword arguments on to its operator
  (e.g. `stable: false`); LTNtorch accepts and ignores them.
- `LTN::Variable` converts any float64 individuals to float32; LTNtorch
  only converts CPU ones.

### Fixed (compared to LTNtorch)

- `SatAgg` reports the shapes of formulas that are not closed; LTNtorch
  raises an unrelated error while building that message.

# Contributing to LTNtorch-rb

This document covers the practical mechanics of working on this codebase: setup, running tests, linting, and coverage. For the pull request process, the Developer Certificate of Origin, and the code of conduct, see the "Contributing" section of `readme.md` and `code_of_conduct.md`.

## Tech stack

This project uses the same, deliberately less-mainstream, tech stack as [reinforce](https://github.com/DSG-UniFE/reinforce) (see `AGENTS.md` for the short version):

- [`sus`](https://github.com/socketry/sus), not RSpec, for testing.
- [`bake`](https://github.com/ioquatix/bake), not Rake, for task running.
- [`standard`](https://github.com/standardrb/standard) for linting/formatting.
- [`covered`](https://github.com/socketry/covered), not SimpleCov, for coverage.
- `gems.rb`, not `Gemfile`. `gems.locked` is intentionally **not** committed: this is a gem, not an application.

## Setup

This project depends on [torch.rb](https://github.com/ankane/torch.rb), which needs a matching LibTorch install to compile its native extension. `.github/workflows/ci.yml` is the source of truth for the exact version (`LIBTORCH_VERSION`); as of this writing, LibTorch 2.15.0 for torch.rb 0.27.x:

```bash
# macOS (Apple Silicon), either via Homebrew...
brew install pytorch
# ...or with the official archive
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.15.0.zip -o libtorch.zip
unzip -q libtorch.zip
bundle config set build.torch-rb --with-torch-dir="$(pwd)/libtorch"

# Linux (CPU)
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.15.0%2Bcpu.zip -o libtorch.zip
unzip -q libtorch.zip
bundle config set build.torch-rb --with-torch-dir="$(pwd)/libtorch"
```

Then install dependencies:

```bash
bundle install
```

## Running tests

```bash
bundle exec sus
```

## Linting

```bash
bundle exec standardrb          # check
bundle exec standardrb --fix    # auto-fix what's safe to auto-fix
```

CI runs this as a blocking check with zero tolerated offenses.

## Coverage

```bash
COVERAGE=BriefSummary bundle exec sus              # overall percentage + least-covered files
COVERAGE=PartialSummary bundle exec sus            # only the uncovered code snippets
bundle exec bake covered:validate --minimum 0.90   # what CI actually checks
```

## Code conventions

- `lib/<path>.rb` and `test/<path>.rb` mirror each other 1:1 (e.g. `lib/ltn/fuzzy_ops.rb` corresponds to `test/ltn/fuzzy_ops.rb`).
- This is a port: keep LTNtorch's semantics, including numerical results, unless a deviation is deliberate. Document every deliberate deviation in the code and in `changelog.md`.
- Ruby idioms replace Python ones where that doesn't change behavior: keyword arguments instead of optional positional ones, `#call` (so `op.(x, y)` works) instead of `__call__`, blocks or lambdas instead of Python `lambda`s, and symbols instead of string flags.

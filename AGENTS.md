# Tech stack

The present project does not use mainstream Ruby gems. Instead it uses:

- the [sus](https://github.com/socketry/sus) gem for testing (instead of, e.g., Rspec)
- the [bake](https://github.com/ioquatix/bake) gem for task management (instead of, e.g., rake)
- the [standard](https://github.com/standardrb/standard) gem for linting/formatting
- the [covered](https://github.com/socketry/covered) gem for coverage (instead of, e.g., SimpleCov) -- wired into `sus` via `config/sus.rb`; run with `COVERAGE=BriefSummary bundle exec sus` locally, or `bundle exec bake covered:validate --minimum <ratio>` to check the persisted `.covered.db` against a threshold (what CI does)
- a gems.rb file instead of a Gemfile

# Porting notes

This library is a port of [LTNtorch](https://github.com/tommasocarraro/LTNtorch) to [torch.rb](https://github.com/ankane/torch.rb). Unless a difference is documented, behavior (including numerical results) should match LTNtorch's: when in doubt, check the Python source.

## torch.rb pitfalls

torch.rb 0.27.0 differs from PyTorch in ways that silently change results (see `TODO_torch-rb.md`):

- `<`, `<=`, `>`, `>=` and `==` on tensors are not element-wise: use `lt`, `le`, `gt`, `ge`, `eq`.
- Write float/tensor arithmetic tensor-first (`x * c`, `Torch.rsub(x, 1.0)` for `1.0 - x`): `Float op Tensor` rounds the float to float32.
- Use `LTN::FuzzyOps.where` instead of `Torch.where`: with 0-dim tensors that do not require grad, `Torch.where` returns float32.
- `Torch.tensor(data, requires_grad: true)` returns a non-leaf tensor that never receives gradients: use `Torch.tensor(data).requires_grad!(true)`.
- `Torch.norm(x, dim: 1)` fails: pass `p` explicitly, `Torch.norm(x, 2, dim: 1)`.
- `Torch.randint` returns float32 (PyTorch: int64): pass `dtype: :int64` for indices or labels. There is no `one_hot`: use `Torch.eye(n)[labels]`.

## LTN pitfalls

- `LTN.diag` changes the labels of its variables, so call it before building the formulas that use them, as in `forall.call(LTN.diag(x, y), p.call(x, y))` (where Ruby evaluates the arguments left to right); a formula built earlier keeps the variables' own labels, and the quantifier leaves them free.

## Examples

`examples/` and `tutorials/` port LTNtorch's. Each example is a class whose `#run(epochs:)` returns its final metrics (`test/examples.rb` runs them briefly, `benchmark/examples.rb` fully); `LTNExamples.main` runs it with LTNtorch's settings when the file is executed. Coverage (`config/covered.rb`) only measures `lib/`.

Reference values come from LTNtorch itself: `test/fixtures/*.py` generate `test/fixtures/*.json` (see their headers), and new behavior should get a fixture case there, float64 and 0-dim variants included.

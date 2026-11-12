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

Reference values come from LTNtorch itself: `test/fixtures/*.py` generate `test/fixtures/*.json` (see their headers), and new behavior should get a fixture case there, float64 and 0-dim variants included.

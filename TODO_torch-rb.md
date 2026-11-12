# TODO: torch-rb fixes needed by LTNtorch-rb

Issues in [torch.rb](https://github.com/ankane/torch.rb) found while porting LTNtorch, to address in our fork (`mtortonesi/torch.rb`, checked out at `~/code/git/torch.rb`) and propose upstream once LTNtorch-rb is done. Found on torch-rb 0.27.0 with LibTorch 2.15.0.

## Status (2026-11-12)

Fixed in `~/code/git/torch.rb`, one local branch per pull request, each a single commit on top of `upstream/master` (0.27.0) with tests and a changelog entry; none pushed yet:

| Branch | Fixes | Notes |
|---|---|---|
| `fix-tensor-requires-grad-leaf` | 1 | Also re-enables `test_max`'s disabled assertion. |
| `fix-scalar-overload-order` | 3 | Reorders 144 of 1,336 functions' overloads; numbers given for Tensor parameters (`x + 2`) become wrapped-number tensors, as in PyTorch, so results do not change. |
| `fix-numeric-coerce` | 4 | |
| `fix-randint-dtype` | 5 (`randint`) | Changes the dtype of existing calls: a minor version, or an issue first. |
| `add-one-hot` | 5 (`one_hot`) | |
| `norm-defaults` | 6 | |
| `fix-list-defaults` | 7, 8 (found while fixing 6) | |

Merged together (local branch `integration-check`, changelog conflicts only), they pass torch.rb's suite (409 tests) and LTNtorch-rb's (650 tests, plus the learning benchmark, with the same results). Without LTNtorch-rb's workarounds (`FuzzyOps.where`, tensor-first arithmetic), its parity tests fail 27 times on 0.27.0 and pass on the merged branches.

Still to do:

- Open the pull requests upstream (`ankane/torch.rb`): the changelog entries all add a `0.27.1 (unreleased)` section, so the second and later ones need rebasing as the first ones land.
- Item 2: open an issue upstream (no code change planned).
- After a release with the fixes: follow "Afterwards in LTNtorch-rb" in each section below.

## 7. List defaults break functions (bug, found while fixing 6)

The codegen copied list defaults from `native_functions.yaml` as is (`dim=[-2,-1]`), but the arg parser expects them in braces (`{-2,-1}`): building the parser threw `stol: no conversion` on the first call of `fft2`, `ifft2`, `rfft2`, `irfft2`, `hfft2`, `ihfft2`, `linalg.matrix_norm` and `rot90`, whatever the arguments. Upstream had disabled tests for some of them. Fixed in `fix-list-defaults`.

## 8. String defaults are never used (bug, found while fixing 6)

`RubyArgs::stringWithDefault` checks for a missing argument with `!args[i]`, as PyTorch does for a null `PyObject*`; but a missing argument is `Qnil` here, which is not 0, so `linalg.qr`, `quantile`, `nanquantile` and `NN.gelu` raised `TypeError` unless given their string argument. Fixed in `fix-list-defaults`.

## 1. `Torch.tensor(data, requires_grad: true)` returns a non-leaf tensor (bug)

### Symptom

```ruby
x = Torch.tensor([1.0, 2.0], requires_grad: true)
x.leaf?            # => false (PyTorch: True)
(x * x).sum.backward
x.grad             # => nil, plus LibTorch's "accessing .grad of a non-leaf Tensor" warning
```

Any data shape is affected, scalars included. The failure is silent apart from the warning: gradients simply never reach the tensor, so an optimizer given such a tensor never updates it. `Torch.zeros/ones/randn/empty(..., requires_grad: true)` are fine, and so is `Torch::NN::Parameter`, which goes through `_make_subclass`.

### Root cause

`make_tensor` in `ext/torch/torch.cpp` (lines 14-24 at upstream commit `37f38f2`) builds a flat tensor with the caller's options, `requires_grad` included, and then reshapes it:

```cpp
torch::Tensor t = torch::tensor(vec, options);
return t.reshape(size);
```

When `requires_grad` is set, `reshape` is a differentiable op, so the returned tensor is a view with a `grad_fn` (a non-leaf). The real leaf is the flat temporary, which nothing references afterwards. torch.rb's own test for this case has its assertion disabled (`test_max` in `test/autograd_test.rb`: `# TODO debug`), probably for this reason.

### Plan

1. Sync the fork with upstream `ankane/torch.rb` (the local checkout is at `122cd9c`, behind upstream's 0.27.0) and branch off, e.g. `fix-tensor-requires-grad-leaf`.
2. Fix `make_tensor`: create the tensor without `requires_grad`, reshape it, then set the flag on the result, which is then a leaf:
   ```cpp
   torch::Tensor t = torch::tensor(vec, options.requires_grad(false)).reshape(size);
   if (options.requires_grad()) {
     t.set_requires_grad(true);
   }
   return t;
   ```
   Check whether LibTorch allows `set_requires_grad` on the output of `reshape` (a view of a tensor that does not require grad, so it should be a leaf). If it does not, use `.detach()` or `.clone()` after the reshape instead.
3. Add regression tests in `test/autograd_test.rb` (Minitest, as in upstream), each asserting `leaf?`, `requires_grad`, and the gradient after `backward`:
   - 1-D, 2-D, and 0-d (scalar) data;
   - `dtype: :float64`;
   - `device:` combined with `requires_grad:` (CUDA/MPS where available);
   - `requires_grad: false` and the default, which must still not require grad.
4. Re-enable the disabled assertion in `test_max` (expected gradient `[0.5, 0, 0.5]`) if the fix makes it pass.
5. Check the other paths that build tensors from Ruby data for the same pattern: `Torch.from_numo`/`_from_blob_ref`, `Tensor.new`/`FloatTensor.new` and friends, and `Torch.tensor` given a Numo array.
6. Run the full torch.rb suite (`bundle exec rake compile && bundle exec rake test`), add a `CHANGELOG.md` entry, and open a pull request upstream with the reproduction above.

### Afterwards in LTNtorch-rb and reinforce

- Once a torch-rb release includes the fix: bump the pin in `gems.rb` (and `LIBTORCH_VERSION` in CI, if that release changes it), then remove `TorchTestHelpers.leaf_tensor` from `test/support/torch_test_helpers.rb` and replace its uses with `Torch.tensor(data, requires_grad: true)`.
- Until then, LTN code must not create learnable tensors with `Torch.tensor(..., requires_grad: true)`, e.g. for trainable `LTN::Constant`s; use `requires_grad!(true)` on the created tensor instead.
- Audit reinforce for `Torch.tensor(..., requires_grad: true)` calls that are expected to receive gradients.

## 2. Comparison operators are not element-wise (limitation, needs discussion upstream)

`Torch::Tensor` includes `Comparable` and defines `<=>` through `item`, so `<`, `<=`, `>`, `>=` and `==` only work on one-element tensors, and raise `only one element tensors can be converted to Ruby scalars` otherwise. In PyTorch they are element-wise. The named methods (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) work, so LTNtorch-rb uses those and does not depend on a change here.

Making the operators element-wise would change torch.rb's public behavior (existing code may rely on `Comparable`, e.g. to sort or compare scalar tensors), so the plan is to open an issue upstream first, not a pull request:

1. Search upstream issues for earlier discussion of this.
2. If there is none, open an issue with the reproduction, the PyTorch behavior, and the trade-off, and offer a pull request if the maintainer agrees.

## 3. 0-dim tensors dispatch to Scalar overloads, losing their dtype (bug)

### Symptom

```ruby
a = Torch.tensor(0.3, dtype: :float64)
b = Torch.tensor(0.6, dtype: :float64)
Torch.where(Torch.tensor(true), a, b).dtype  # => :float32 (PyTorch: float64)
Torch.where(Torch.tensor(true), 1.0, b).dtype # => :float32 (PyTorch: float64)
```

Only 0-dim tensors that do not require grad are affected; with any other shape, or with `requires_grad`, the result is right. In LTN this hits closed formulas (scalars) computed by guarded quantification, which are float64, whenever they are combined further without gradients (e.g. at evaluation time).

### Root cause

torch.rb's argument parser (`ext/torch/ruby_arg_parser.cpp`, `FunctionParameter::check`, `case ParameterType::SCALAR`) accepts a 0-dim tensor that does not require grad for a `Scalar` parameter, as PyTorch's does. But PyTorch also orders each function's overloads so that `Tensor` signatures are tried before `Scalar` ones (`sort_overloads` in torchgen's `gen_python_functions.py`), while torch.rb's codegen (`sort_functions` in `codegen/generate_functions.rb`) only moves `out` variants last. So for `where`, `where.Scalar(Tensor condition, Scalar self, Scalar other)` can match first, and computes in the default dtype.

### Plan

1. Write a failing test with the reproduction above, plus the same check for other functions with both Tensor and Scalar overloads (`sub`, `add`, `mul`, `div`, `pow`, `masked_fill`, `clamp`, `where`).
2. Port torchgen's overload ordering to `sort_functions`: for two signatures that differ only in Tensor vs Scalar parameters, the Tensor one goes first. Regenerate the bindings and rebuild.
3. Run the full suite, and open an upstream pull request.

### Afterwards in LTNtorch-rb

- Remove `LTN::FuzzyOps.where`, using `Torch.where` directly (`lib/ltn/fuzzy_ops.rb`, `lib/ltn/quantifier.rb`), keeping the float64 cases in `test/fixtures/fuzzy_ops.py` as regression tests.

## 4. `Float op Tensor` converts the float to a float32 tensor (bug)

### Symptom

```ruby
v = Torch.tensor([0.3, 0.7], dtype: :float64)
((1 - 1e-4) * v).to_a # => [0.29996999502182004, 0.6999299883842468] (PyTorch: [0.29997, 0.69993])
(v * (1 - 1e-4)).to_a # => [0.29997, 0.6999299999999999]
(1.0 - Torch.tensor(0.3, dtype: :float64)).dtype # => :float32 (PyTorch: float64)
```

### Root cause

`Tensor#coerce` (`lib/torch/tensor.rb`) returns `[Torch.tensor(other), self]`: the Ruby float becomes a float32 tensor, so its value is rounded to float32 before the operation. PyTorch treats a Python number as a "wrapped number" instead, which takes part in the operation in the dtype of the tensor. With a 0-dim tensor, the result is also float32, because of issue 3.

### Plan

1. Write failing tests for `Float` and `Integer` with `+`, `-`, `*`, `/` and `**` on float32, float64 and integer tensors, comparing values and dtypes with PyTorch's.
2. Make `coerce` return a wrapper for the number whose operators call the tensor's methods with the number as a Scalar (`+`/`*` commute; `-` is `Torch.rsub(tensor, number)`; `/` is `Torch.reciprocal(tensor) * number` or `Torch.div(Torch.full_like(tensor, number), tensor)`; `**` is `Torch.pow(number, tensor)`), so the number keeps PyTorch's wrapped-number semantics.
3. Run the full suite, and open an upstream pull request.

### Afterwards in LTNtorch-rb

- Nothing is required: LTN writes float/tensor arithmetic tensor-first (`x * c`, `Torch.rsub(x, 1.0)`), which is correct either way. Optionally restore the more readable `1.0 - x` forms.

## 5. Smaller API differences

- `Torch.randint(low, high, size)` returns float32 tensors; PyTorch returns int64. LTN's examples pass `dtype: :int64`. Check whether this is deliberate upstream (torch.rb may default every factory to float32) before proposing a change.
- `torch.nn.functional.one_hot` has no counterpart (`Torch::NN::Functional.one_hot` and `Torch.one_hot` are undefined); `Torch.eye(n)[labels]` works around it. Plan: add `Torch::NN::Functional.one_hot(tensor, num_classes: -1)`, binding the native `one_hot`, with tests.

## 6. `Torch.norm(x, dim: 1)` needs an explicit `p` (API gap)

`Torch.norm(x, dim: 1)` raises `norm() got an unexpected keyword argument 'dim'`; `Torch.norm(x, 2, dim: 1)` works. In PyTorch, `torch.norm` is a Python wrapper (`torch/functional.py`) that defaults `p` to `"fro"` and dispatches to the native overloads, whose `p` has no default when `dim` is given. torch.rb exposes the native overloads directly.

Plan: add a Ruby-side `Torch.norm`/`Tensor#norm` wrapper with PyTorch's defaults (`p: "fro"`, `dim: nil`, `keepdim: false`, `dtype: nil`), with tests, or open an issue first to agree on the API, since `p` is positional in the current one.

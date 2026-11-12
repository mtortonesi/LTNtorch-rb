# TODO: torch-rb fixes needed by LTNtorch-rb

Issues in [torch.rb](https://github.com/ankane/torch.rb) found while porting LTNtorch, to address in our fork (`mtortonesi/torch.rb`, checked out at `~/code/git/torch.rb`) and propose upstream once LTNtorch-rb is done. Found on torch-rb 0.27.0 with LibTorch 2.15.0.

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

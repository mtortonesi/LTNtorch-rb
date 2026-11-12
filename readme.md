# LTNtorch-rb

An implementation of [Logic Tensor Networks](https://arxiv.org/abs/2012.13635)
(LTN) on top of [torch.rb](https://github.com/ankane/torch.rb), ported from
[LTNtorch](https://github.com/tommasocarraro/LTNtorch), the PyTorch
implementation by Tommaso Carraro. It is developed at the [Big Data and
Compute Continuum Research Lab](https://bdcc.unife.it) of the University of
Ferrara, Italy, as a companion to the
[reinforce](https://github.com/DSG-UniFE/reinforce) Reinforcement Learning
library.

LTN is a neuro-symbolic framework: predicates, functions and constants of a
first-order logic language are grounded as (possibly learnable) tensors and
neural networks, and the axioms of a knowledge base are evaluated with
differentiable fuzzy logic semantics. Maximizing the satisfaction of the
axioms becomes the training objective.

**Status:** under active development. The core of LTN (groundings,
predicates, functions, connectives, quantifiers, guarded and diagonal
quantification, and the fuzzy operators) is ported, and checked against
LTNtorch's results; see `changelog.md` for progress and for the
differences from LTNtorch.

## Prerequisites

LTNtorch-rb requires the [torch.rb](https://rubygems.org/gems/torch-rb) gem,
which provides Ruby bindings for LibTorch (the C++ core of PyTorch). torch.rb
compiles against exactly one LibTorch version (0.27.x needs LibTorch 2.15.x).

On macOS you can install LibTorch with Homebrew:

    $ brew install pytorch

Otherwise, download LibTorch from [the PyTorch Web site](https://pytorch.org/),
extract it, and point Bundler at it:

    $ bundle config build.torch-rb --with-torch-dir=/path/to/libtorch

For more info, see [torch.rb's readme](https://github.com/ankane/torch.rb).

## Installation

Add the gem to your application's gems.rb (or Gemfile):

    $ bundle add ltntorch

or install it directly:

    $ gem install ltntorch

Then:

```ruby
require "ltn"
```

## Usage

Learn a predicate `A` from a knowledge base saying that it holds for one
cluster of points and not for another:

```ruby
require "ltn"

positives = Torch.randn(50, 2) * 0.5 + 1.0
negatives = Torch.randn(50, 2) * 0.5 - 1.0

# A learnable predicate: any Torch::NN::Module returning truth values in [0, 1].
a = LTN::Predicate.new(
  Torch::NN::Sequential.new(
    Torch::NN::Linear.new(2, 8), Torch::NN::ELU.new,
    Torch::NN::Linear.new(8, 1), Torch::NN::Sigmoid.new
  )
)

not_ = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
sat_agg = LTN::FuzzyOps::SatAgg.new

x_pos = LTN::Variable.new("x_pos", positives)
x_neg = LTN::Variable.new("x_neg", negatives)
optimizer = Torch::Optim::Adam.new(a.parameters, lr: 0.01)

200.times do
  optimizer.zero_grad
  # forall x_pos: A(x_pos), and forall x_neg: not A(x_neg)
  sat = sat_agg.call(
    forall.call(x_pos, a.call(x_pos)),
    forall.call(x_neg, not_.call(a.call(x_neg)))
  )
  (1.0 - sat).backward
  optimizer.step
end
```

Predicates and functions can also be grounded by a block, e.g.
`LTN::Predicate.new { |x| Torch.sigmoid(x.sum(1)) }`, and quantifiers can be
guarded by a condition on the individuals:

```ruby
forall.call(x, a.call(x), cond_vars: [x]) { |x| x.value.sum(1).gt(0.0) }
```

## Examples and tutorials

`examples/` contains Ruby ports of LTNtorch's seven examples, and
`tutorials/` of its four tutorials, as scripts:

| Script | What it shows |
|---|---|
| `tutorials/1_grounding_non_logical_symbols.rb` | Constants, predicates, functions and variables |
| `tutorials/2_grounding_connectives.rb` | Connectives, quantifiers, diagonal and guarded quantification |
| `tutorials/2b_operators_and_gradients.rb` | Choosing fuzzy operators, and their gradients |
| `tutorials/3_knowledgebase_and_learning.rb` | Learning from a knowledge base, in mini-batches too |
| `examples/1_binary_classification.rb` | A predicate learned from positive and negative examples |
| `examples/2_multi_class_single_label_classification.rb` | Iris, with a softmax classifier as predicate |
| `examples/3_multi_class_multi_label_classification.rb` | Crabs, with mutually exclusive labels, and queries |
| `examples/4_semi_supervised_pattern_recognition.rb` | MNIST digits learned from their sums only (downloads MNIST) |
| `examples/5_regression.rb` | Real estate prices, with a smooth equality and diagonal quantification |
| `examples/6_clustering.rb` | Clustering without labels, with guarded quantification |
| `examples/7_learning_embeddings_with_ltn.rb` | Smokers, friends and cancer: learned embeddings of individuals |

Run them with, e.g., `bundle exec ruby examples/6_clustering.rb`; set
`EPOCHS` to train for fewer epochs. Their results are close to those of
LTNtorch's notebooks, and `bundle exec sus benchmark/examples.rb` checks
that they keep learning.

The API follows LTNtorch's, so its
[documentation](https://tommasocarraro.github.io/LTNtorch/) and
[tutorials](https://github.com/tommasocarraro/LTNtorch/tree/main/tutorials)
apply, with Ruby conventions: optional arguments are keywords, symbols are
called with `#call`, and quantifiers take `:forall`/`:exists`.

## Contributing

See `CONTRIBUTING.md` for setup, tests, linting and coverage.

1.  Fork it.
2.  Create your feature branch (`git checkout -b my-new-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin my-new-feature`).
5.  Create new Pull Request.

## Credits and citation

LTNtorch-rb is a port of LTNtorch; if you use it in your work, please also
cite the original software and the LTN paper:

- T. Carraro, *LTNtorch: PyTorch implementation of Logic Tensor Networks*,
  [doi:10.5281/zenodo.6394282](https://doi.org/10.5281/zenodo.6394282).
- S. Badreddine, A. d'Avila Garcez, L. Serafini, M. Spranger, *Logic Tensor
  Networks*, Artificial Intelligence 303, 2022,
  [arXiv:2012.13635](https://arxiv.org/abs/2012.13635).

## License

This software is available as open source under the terms of the [MIT
License](https://opensource.org/licenses/MIT); see `license.md`.

### Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). All contributors to this project must agree to this document to have their contributions accepted.

### Contributor Covenant

This project is governed by [Contributor Covenant](https://www.contributor-covenant.org/). All contributors and participants agree to abide by its terms.

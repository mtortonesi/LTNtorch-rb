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

**Status:** under active development. The port is not functional yet; see
`changelog.md` for progress.

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

# frozen_string_literal: true

require "torch"

module TorchTestHelpers
  module_function

  DEFAULT_SEED = 12_345

  def with_torch_seed(seed = test_seed)
    srand(seed)
    Torch.manual_seed(seed)
    yield(seed)
  end

  def test_seed
    Integer(ENV.fetch("LTN_TEST_SEED", DEFAULT_SEED.to_s))
  end

  # A leaf tensor that requires grad, so backward populates its #grad.
  #
  # Works around a torch.rb 0.27.0 bug: `Torch.tensor(data, requires_grad: true)`
  # returns a non-leaf tensor (a reshape of the leaf it creates natively), whose
  # #grad stays nil after backward.
  def leaf_tensor(data, **options)
    Torch.tensor(data, **options).requires_grad!(true)
  end

  # Element-wise comparison of two tensors (or nested arrays) within an
  # absolute tolerance, for checking results against reference values.
  def all_close?(actual, expected, tolerance: 1e-6)
    actual = actual.to_a if actual.respond_to?(:to_a)
    expected = expected.to_a if expected.respond_to?(:to_a)
    actual = Array(actual).flatten
    expected = Array(expected).flatten
    actual.size == expected.size &&
      actual.zip(expected).all? { |a, e| (a - e).abs <= tolerance }
  end
end

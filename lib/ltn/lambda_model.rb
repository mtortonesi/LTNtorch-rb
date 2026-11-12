# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # Wraps a function (a lambda, proc, method or block) in a Torch::NN::Module,
  # to ground a non-trainable predicate or function.
  class LambdaModel < Torch::NN::Module
    # @attribute [#call] the wrapped function.
    attr_reader :func

    # @parameter func [#call] the function to wrap.
    def initialize(func)
      super()
      @func = func
    end

    def forward(*inputs, **kwargs)
      func.call(*inputs, **kwargs)
    end
  end
end

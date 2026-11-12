# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN constant: an individual of the domain, grounded as a tensor with no
  # free variables. A trainable constant is an embedding learned along with
  # the predicates and functions of the knowledge base.
  class Constant < LTNObject
    # @parameter value [Torch::Tensor] the grounding, moved to LTN.device.
    # @parameter trainable [Boolean] whether to learn the grounding. A
    #   trainable constant is converted to float32 and requires grad.
    # @raises [TypeError] if `value` is not a tensor.
    def initialize(value, trainable: false)
      super(value, [])
      @value = @value.to(LTN.device)
      return unless trainable

      @value = @value.float
      @value.requires_grad!(true)
    end

    def inspect
      "Constant(value=#{value.inspect}, free_vars=#{free_vars.inspect})"
    end
    alias_method :to_s, :inspect
  end
end

# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # The output of every LTN operation: a grounding (a tensor) together with
  # the labels of the free variables appearing in it, one per leading axis of
  # the tensor, in order.
  #
  # Named `LTNObject`, as in LTNtorch, rather than `LTN::Object`, which would
  # shadow Ruby's `Object` for every unqualified reference inside `LTN`.
  class LTNObject
    # @attribute [Torch::Tensor] the grounding.
    attr_accessor :value

    # @attribute [Array(String)] labels of the free variables, one per leading axis of #value.
    attr_accessor :free_vars

    # @parameter value [Torch::Tensor] the grounding.
    # @parameter var_labels [Array(String)] labels of the free variables in `value`.
    # @raises [TypeError] if `value` is not a tensor or `var_labels` is not an array of strings.
    def initialize(value, var_labels)
      unless value.is_a?(Torch::Tensor)
        raise TypeError, "LTNObject.new: argument 'value' (position 1) must be a Torch::Tensor, not #{value.class}"
      end
      unless var_labels.is_a?(Array) && var_labels.all?(String)
        raise TypeError,
          "LTNObject.new: argument 'var_labels' (position 2) must be an Array of String, not #{var_labels.class}"
      end

      @value = value
      @free_vars = var_labels
    end

    # @returns [Array(Integer)] the shape of the grounding.
    def shape
      value.shape
    end

    def inspect
      "LTNObject(value=#{value.inspect}, free_vars=#{free_vars.inspect})"
    end
    alias_method :to_s, :inspect
  end
end

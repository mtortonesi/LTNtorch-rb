# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN connective: applies a fuzzy connective operator (see LTN::FuzzyOps)
  # to LTN objects, after broadcasting them (see LTN.process_ltn_objects).
  #
  #   and_ = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
  #   and_.call(p.call(x), q.call(y)) # free_vars ["x", "y"]
  class Connective
    # @attribute [FuzzyOps::ConnectiveOperator] the operator.
    attr_reader :connective_op

    # @parameter connective_op [FuzzyOps::ConnectiveOperator] the operator.
    # @raises [TypeError] if `connective_op` is not a connective operator.
    def initialize(connective_op)
      unless connective_op.is_a?(FuzzyOps::ConnectiveOperator)
        raise TypeError, "Connective.new: argument 'connective_op' (position 1) must be a " \
          "LTN::FuzzyOps::ConnectiveOperator, not #{connective_op.class}"
      end

      @connective_op = connective_op
    end

    # @parameter operands [Array(LTNObject)] one operand for a unary operator, two for a binary one.
    # @parameter kwargs [Hash] passed to the operator, e.g. `stable: false`. (LTNtorch ignores them.)
    # @returns [LTNObject] the result, with one axis per free variable of the operands.
    # @raises [ArgumentError] if the number of operands does not match the
    #   operator, or their values are not in [0, 1].
    # @raises [TypeError] if an operand is not an LTNObject.
    def call(*operands, **kwargs)
      if connective_op.is_a?(FuzzyOps::UnaryConnectiveOperator) && operands.size != 1
        raise ArgumentError, "Expected one operand for a unary connective, but got #{operands.size}"
      end
      if connective_op.is_a?(FuzzyOps::BinaryConnectiveOperator) && operands.size != 2
        raise ArgumentError, "Expected two operands for a binary connective, but got #{operands.size}"
      end
      unless operands.all?(LTNObject)
        raise TypeError, "Expected parameter 'operands' to be LTNObjects, but got #{operands.map(&:class).inspect}"
      end

      FuzzyOps.check_values(*operands.map(&:value))
      processed, vars, n_individuals_per_var = LTN.process_ltn_objects(operands)
      output = connective_op.call(*processed.map(&:value), **kwargs)
      LTNObject.new(output.reshape(n_individuals_per_var), vars)
    end

    def inspect
      "Connective(connective_op=#{connective_op.inspect})"
    end
    alias_method :to_s, :inspect
  end
end

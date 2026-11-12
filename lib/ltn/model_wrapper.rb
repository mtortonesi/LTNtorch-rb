# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # The construction shared by Predicate and Function: they are grounded
  # either by a Torch::NN::Module (trainable) or by a function (not trainable).
  module ModelWrapper
    # @attribute [Torch::NN::Module] the grounding; a LambdaModel for a function.
    attr_reader :model

    # Ground the symbol with either `model` or a function, given as `func:`
    # or as a block:
    #
    #   LTN::Predicate.new(model)
    #   LTN::Predicate.new(func: ->(x) { Torch.sigmoid(x.sum(1)) })
    #   LTN::Predicate.new { |x| Torch.sigmoid(x.sum(1)) }
    #
    # @parameter model [Torch::NN::Module, nil] a model taking one tensor per argument.
    # @parameter func [#call, nil] a function taking one tensor per argument.
    # @raises [ArgumentError] unless exactly one of model, func and block is given.
    # @raises [TypeError] if `model` is not a Torch::NN::Module or `func` is not callable.
    def initialize(model = nil, func: nil, &block)
      super()
      groundings = [model, func, block].compact
      if groundings.size > 1
        raise ArgumentError, "More than one of model, func and block have been specified. " \
          "Expected only one of them to be specified."
      end
      if groundings.empty?
        raise ArgumentError, "None of model, func and block have been specified. " \
          "Expected one of them to be specified."
      end

      @model = if model
        unless model.is_a?(Torch::NN::Module)
          raise TypeError, "#{symbol_name}.new: argument 'model' (position 1) must be a Torch::NN::Module, " \
            "not #{model.class}"
        end
        model
      else
        func ||= block
        if !func.respond_to?(:call) || func.is_a?(Torch::NN::Module)
          raise TypeError, "#{symbol_name}.new: argument 'func' must be a function, not #{func.class}"
        end
        LambdaModel.new(func)
      end
    end

    def inspect
      "#{symbol_name}(model=#{model.inspect})"
    end

    def to_s
      inspect
    end

    private

    def symbol_name
      self.class.name.split("::").last
    end

    def check_inputs(inputs)
      return if inputs.all?(LTNObject)

      raise TypeError, "Expected parameter 'inputs' to be LTNObjects, but got #{inputs.map(&:class).inspect}"
    end
  end
end

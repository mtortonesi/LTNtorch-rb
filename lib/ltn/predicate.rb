# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN predicate: maps individuals to truth values in [0, 1].
  #
  #   is_positive = LTN::Predicate.new { |x| Torch.sigmoid(x.sum(1)) }
  #   x = LTN::Variable.new("x", Torch.randn(4, 3))
  #   is_positive.call(x) # => an LTNObject with 4 truth values and free_vars ["x"]
  class Predicate < Torch::NN::Module
    include ModelWrapper

    # Computes the truth values of the predicate, after broadcasting the
    # inputs (see LTN.process_ltn_objects): the result has one axis per free
    # variable of the inputs.
    #
    # @parameter inputs [Array(LTNObject)] the arguments.
    # @parameter kwargs [Hash] passed to the model.
    # @returns [LTNObject]
    # @raises [TypeError] if an input is not an LTNObject.
    # @raises [ArgumentError] if the model returns values outside [0, 1].
    def forward(*inputs, **kwargs)
      check_inputs(inputs)
      processed, output_vars, output_shape = LTN.process_ltn_objects(inputs)
      output = model.call(*processed.map(&:value), **kwargs)

      unless Torch.all(Torch.logical_and(output.ge(0.0), output.le(1.0))).item
        raise ArgumentError, "Expected the output of a predicate to be in the range [0., 1.], but got some " \
          "values outside of this range. Check your predicate implementation!"
      end

      LTNObject.new(output.reshape(output_shape).float, output_vars)
    end
  end
end

# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN function: maps individuals to individuals (tensors of any shape).
  class Function < Torch::NN::Module
    include ModelWrapper

    # Computes the function, after broadcasting the inputs (see
    # LTN.process_ltn_objects): the result has one leading axis per free
    # variable of the inputs, followed by the dimensions of each output.
    #
    # @parameter inputs [Array(LTNObject)] the arguments.
    # @parameter kwargs [Hash] passed to the model.
    # @returns [LTNObject]
    # @raises [TypeError] if an input is not an LTNObject.
    def forward(*inputs, **kwargs)
      check_inputs(inputs)
      processed, output_vars, output_shape = LTN.process_ltn_objects(inputs)
      output = model.call(*processed.map(&:value), **kwargs)

      LTNObject.new(output.reshape(output_shape + output.shape.drop(1)).float, output_vars)
    end
  end
end

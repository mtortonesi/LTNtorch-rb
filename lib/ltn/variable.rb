# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN variable: a sequence of individuals of the domain, grounded as a
  # tensor whose first dimension indexes the individuals.
  class Variable < LTNObject
    # @attribute [String] the label given to the variable, which #free_vars
    #   replaces temporarily during diagonal quantification (see LTN.diag).
    attr_reader :latent_var

    # @parameter var_label [String] the label of the variable. Labels starting
    #   with `diag_` are reserved for diagonal quantification.
    # @parameter individuals [Torch::Tensor] the individuals, one per row.
    #   float64 values are converted to float32.
    # @parameter add_batch_dim [Boolean] whether to turn a 1-D tensor of n
    #   individuals into a n x 1 tensor, so that each individual is a vector.
    # @raises [TypeError] if the label is not a String or the individuals are not a tensor.
    # @raises [ArgumentError] if the label starts with `diag_`.
    def initialize(var_label, individuals, add_batch_dim: true)
      unless var_label.is_a?(String)
        raise TypeError, "Variable.new: argument 'var_label' (position 1) must be a String, not #{var_label.class}"
      end
      if var_label.start_with?("diag_")
        raise ArgumentError, "Labels starting with 'diag_' are reserved for diagonal quantification."
      end
      unless individuals.is_a?(Torch::Tensor)
        raise TypeError, "Variable.new: argument 'individuals' (position 2) must be a Torch::Tensor, not #{individuals.class}"
      end

      super(individuals, [var_label])

      # Avoids dtype mismatches with float32 models. (LTNtorch converts CPU
      # float64 tensors only; here any float64 tensor is converted.)
      @value = @value.float if @value.dtype == :float64
      @value = @value.view(@value.shape[0], 1) if @value.dim == 1 && add_batch_dim
      @value = @value.to(LTN.device)
      @latent_var = var_label
    end

    def inspect
      "Variable(value=#{value.inspect}, free_vars=#{free_vars.inspect})"
    end
    alias_method :to_s, :inspect
  end
end

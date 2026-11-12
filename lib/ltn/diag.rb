# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # Sets diagonal quantification on the given variables: they are no longer
  # broadcast against each other but paired element-wise, the i-th individual
  # of each with the i-th of the others. A quantifier over the variables
  # resets them (see LTN.undiag).
  #
  #   forall.call(LTN.diag(x, y), formula) # for all i, formula(x_i, y_i)
  #
  # This modifies the variables, replacing their free variables with a shared
  # `diag_...` label.
  #
  # @parameter vars [Array(Variable)] at least two variables with the same number of individuals.
  # @returns [Array(Variable)] the same variables.
  # @raises [TypeError] if an argument is not a Variable.
  # @raises [ArgumentError] if there is only one variable, or their numbers of individuals differ.
  def self.diag(*vars)
    unless vars.all?(Variable)
      raise TypeError, "Expected parameter 'vars' to be Variables, but got #{vars.map(&:class).inspect}"
    end
    if vars.size < 2
      raise ArgumentError, "Expected parameter 'vars' to be more than one Variable, but got just one Variable."
    end
    n_individuals = vars.map { |var| var.shape[0] }
    if n_individuals.uniq.size != 1
      raise ArgumentError, "Expected the given LTN variables to have the same number of individuals, " \
        "but got the following numbers of individuals #{n_individuals.inspect}"
    end

    diag_label = "diag_" + vars.map(&:latent_var).join("_")
    vars.each { |var| var.free_vars = [diag_label] }
    vars
  end

  # Resets the given variables from diagonal quantification (see LTN.diag)
  # to their own labels.
  #
  # @parameter vars [Array(Variable)] the variables.
  # @returns [Array(Variable)] the same variables.
  # @raises [TypeError] if an argument is not a Variable.
  def self.undiag(*vars)
    unless vars.all?(Variable)
      raise TypeError, "Expected parameter 'vars' to be Variables, but got #{vars.map(&:class).inspect}"
    end

    vars.each { |var| var.free_vars = [var.latent_var] }
    vars
  end
end

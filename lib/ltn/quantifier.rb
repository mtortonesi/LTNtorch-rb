# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # An LTN quantifier: aggregates the truth values of a formula along the axes
  # of the quantified variables, with an aggregation operator (see
  # LTN::FuzzyOps).
  #
  #   forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
  #   exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists)
  #   forall.call(x, exists.call(y, p.call(x, y)))
  #
  # Guarded quantification restricts the aggregation to the individuals that
  # satisfy a condition:
  #
  #   forall.call(x, p.call(x), cond_vars: [x]) { |x| x.value.sum(1).gt(0.5) }
  class Quantifier
    QUANTIFIERS = {:forall => :forall, :exists => :exists, "f" => :forall, "e" => :exists}.freeze

    # @attribute [FuzzyOps::AggregationOperator] the aggregation operator.
    attr_reader :agg_op

    # @attribute [Symbol] :forall or :exists.
    attr_reader :quantifier

    # @parameter agg_op [FuzzyOps::AggregationOperator] the aggregation operator.
    # @parameter quantifier [Symbol, String] :forall or :exists (LTNtorch's "f" and "e" work too).
    #   It determines the truth value of a guarded quantification over no individuals:
    #   1 for :forall, 0 for :exists.
    # @raises [TypeError] if `agg_op` is not an aggregation operator.
    # @raises [ArgumentError] if `quantifier` is not one of the above.
    def initialize(agg_op, quantifier)
      unless agg_op.is_a?(FuzzyOps::AggregationOperator)
        raise TypeError, "Quantifier.new: argument 'agg_op' (position 1) must be a " \
          "LTN::FuzzyOps::AggregationOperator, not #{agg_op.class}"
      end
      unless QUANTIFIERS.key?(quantifier)
        raise ArgumentError, "Expected parameter 'quantifier' to be :exists (or 'e') for the existential " \
          "quantifier, or :forall (or 'f') for the universal one, but got #{quantifier.inspect}"
      end

      @agg_op = agg_op
      @quantifier = QUANTIFIERS.fetch(quantifier)
    end

    # Quantifies `formula` over `vars`, then resets any diagonal
    # quantification on them (see LTN.diag).
    #
    # @parameter vars [Variable, Array(Variable)] the quantified variables.
    # @parameter formula [LTNObject] the formula.
    # @parameter cond_vars [Variable, Array(Variable), nil] the variables of
    #   the guard, for guarded quantification.
    # @parameter cond_fn [#call, nil] the guard: given the (broadcast) LTN
    #   objects of `cond_vars`, returns a boolean tensor with one value per
    #   combination of their individuals. It can be given as a block instead.
    # @parameter kwargs [Hash] passed to the aggregation operator, e.g. `p: 4`.
    # @returns [LTNObject] the result, whose free variables are those of the
    #   formula minus the quantified ones. Guarded quantification returns
    #   float64 values, as in LTNtorch.
    # @raises [ArgumentError] if only one of `cond_vars` and `cond_fn` is
    #   given, a quantified variable does not appear in the formula, or the
    #   formula's values are not in [0, 1].
    # @raises [TypeError] if an argument has the wrong type.
    def call(vars, formula, cond_vars: nil, cond_fn: nil, **kwargs, &block)
      if cond_fn && block
        raise ArgumentError, "Expected the guard as either 'cond_fn' or a block, but got both."
      end
      cond_fn ||= block
      if cond_fn && cond_vars.nil?
        raise ArgumentError, "Since 'cond_fn' has been set, 'cond_vars' must be set as well, but got nil."
      end
      if cond_vars && cond_fn.nil?
        raise ArgumentError, "Since 'cond_vars' has been set, 'cond_fn' must be set as well, but got nil."
      end
      vars = variables(vars, "vars")
      unless formula.is_a?(LTNObject)
        raise TypeError, "Expected parameter 'formula' to be an LTNObject, but got #{formula.class}"
      end

      FuzzyOps.check_values(formula.value)

      aggregation_vars = vars.map { |var| var.free_vars.first }.uniq
      if cond_fn
        cond_vars = variables(cond_vars, "cond_vars")
        unless cond_fn.respond_to?(:call)
          raise TypeError, "Expected parameter 'cond_fn' to be a function, but got #{cond_fn.class}"
        end

        formula, mask = self.class.compute_mask(formula, cond_vars, cond_fn, aggregation_vars)
        output = agg_op.call(formula.value, dim: aggregation_dims(formula, aggregation_vars), mask: mask.value, **kwargs)
        # The aggregation is undefined (NaN) where the guard leaves no
        # individuals: forall is then vacuously true, exists false.
        output = FuzzyOps.where(Torch.isnan(output), (quantifier == :forall) ? 1.0 : 0.0, output.double)
      else
        output = agg_op.call(formula.value, dim: aggregation_dims(formula, aggregation_vars), **kwargs)
      end

      LTN.undiag(*vars)
      LTNObject.new(output, formula.free_vars - aggregation_vars)
    end

    # Prepares a guarded quantification: broadcasts `formula` over the
    # condition variables it lacks, and computes the mask of `cond_fn`. Both
    # are transposed so that their leading axes are the condition variables
    # that are not quantified, then the quantified ones.
    #
    # Unlike LTNtorch, this does not modify `formula`.
    #
    # @parameter formula [LTNObject] the formula.
    # @parameter cond_vars [Array(Variable)] the variables of the guard.
    # @parameter cond_fn [#call] the guard.
    # @parameter aggregation_vars [Array(String)] labels of the quantified variables.
    # @returns [Array(LTNObject)] the formula and the mask, with the same shape and free variables.
    def self.compute_mask(formula, cond_vars, cond_fn, aggregation_vars)
      cond_vars_not_in_formula = cond_vars.reject { |var| formula.free_vars.include?(var.free_vars.first) }
      unless cond_vars_not_in_formula.empty?
        processed, _, n_individuals = LTN.process_ltn_objects([formula] + cond_vars_not_in_formula)
        formula = LTNObject.new(processed.first.value.view(n_individuals), processed.first.free_vars)
      end

      cond_labels = cond_vars.map { |var| var.free_vars.first }
      cond_not_aggregated = cond_labels.reject { |var| aggregation_vars.include?(var) }
      cond_aggregated = cond_labels.select { |var| aggregation_vars.include?(var) }
      not_in_cond = formula.free_vars.reject { |var| cond_labels.include?(var) }
      formula = transpose_vars(formula, (cond_not_aggregated + cond_aggregated + not_in_cond).uniq)

      processed_cond_vars, cond_vars_order, n_individuals_per_var = LTN.process_ltn_objects(cond_vars)
      mask = cond_fn.call(*processed_cond_vars)
      mask = LTNObject.new(mask.reshape(n_individuals_per_var), cond_vars_order)
      mask = transpose_vars(mask, (cond_not_aggregated + cond_aggregated).uniq)

      if formula.shape != mask.shape
        (formula, mask), _, n_individuals_per_var = LTN.process_ltn_objects([formula, mask])
        mask = LTNObject.new(mask.value.view(n_individuals_per_var), mask.free_vars)
        formula = LTNObject.new(formula.value.view(n_individuals_per_var), formula.free_vars)
      end

      [formula, mask]
    end

    # @parameter object [LTNObject] the object to transpose.
    # @parameter new_vars_order [Array(String)] its free variables, in the new order.
    # @returns [LTNObject] a new object with the axes of the free variables permuted.
    #   (LTNtorch modifies and returns `object`.)
    def self.transpose_vars(object, new_vars_order)
      permutation = new_vars_order.map { |var| object.free_vars.index(var) }
      LTNObject.new(object.value.permute(permutation), new_vars_order)
    end

    def inspect
      "Quantifier(agg_op=#{agg_op.inspect}, quantifier=#{quantifier.inspect})"
    end
    alias_method :to_s, :inspect

    private

    def variables(vars, name)
      return [vars] if vars.is_a?(Variable)
      return vars if vars.is_a?(Array) && vars.all?(Variable)

      types = vars.is_a?(Array) ? vars.map(&:class).inspect : vars.class
      raise TypeError, "Expected parameter '#{name}' to be a Variable or an Array of Variable, but got #{types}"
    end

    def aggregation_dims(formula, aggregation_vars)
      aggregation_vars.map do |var|
        formula.free_vars.index(var) ||
          raise(ArgumentError, "Variable #{var.inspect} does not appear in the formula (free variables: " \
            "#{formula.free_vars.inspect})")
      end
    end
  end
end

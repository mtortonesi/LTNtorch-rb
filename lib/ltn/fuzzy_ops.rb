# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # Fuzzy logic operators and aggregators, the building blocks of
  # LTN::Connective and LTN::Quantifier. See the appendix of the LTN paper
  # (https://arxiv.org/abs/2012.13635) for their definitions.
  #
  # All operators work on tensors of truth values and support torch's usual
  # broadcasting. Each one responds to #call, so it can be invoked as
  # `op.(x, y)`.
  #
  # Differences from LTNtorch's `ltn.fuzzy_ops`:
  # - optional arguments are keywords: `AndProd.new(stable: false)`,
  #   `op.(x, y, stable: false)`, `agg.(xs, dim: [0], mask: m, p: 3)`;
  # - Python's ValueError becomes ArgumentError;
  # - SatAgg reports the shapes of non-scalar formulas instead of raising an
  #   unrelated error while building that message.
  module FuzzyOps
    # The epsilon of the projections #pi_0 and #pi_1, which keep the stable
    # operators away from the regions where their gradients vanish or explode.
    EPS = 1e-4

    module_function

    # Maps truth values from [0, 1] to (0, 1], so that 0 is never an input.
    #
    # @parameter x [Torch::Tensor] truth values.
    # @returns [Torch::Tensor]
    def pi_0(x)
      (1 - EPS) * x + EPS
    end

    # Maps truth values from [0, 1] to [0, 1), so that 1 is never an input.
    #
    # @parameter x [Torch::Tensor] truth values.
    # @returns [Torch::Tensor]
    def pi_1(x)
      (1 - EPS) * x
    end

    # @parameter values [Array(Torch::Tensor)] truth values.
    # @raises [ArgumentError] if any value is outside [0, 1] (or NaN).
    def check_values(*values)
      values.each do |v|
        next if Torch.all(Torch.logical_and(v.ge(0.0), v.le(1.0))).item

        raise ArgumentError, "Expected inputs of connectives and quantifiers to be tensors of truth values " \
          "in the range [0., 1.], but got some values outside this range."
      end
    end

    # Checks the mask of a guarded quantification.
    #
    # @parameter mask [Torch::Tensor] the mask: true includes a value, false excludes it.
    # @parameter xs [Torch::Tensor] the grounding of the formula being aggregated.
    # @raises [ArgumentError] if the shapes differ or the mask is not boolean.
    def check_mask(mask, xs)
      raise ArgumentError, "'xs' and 'mask' must have the same shape." if mask.shape != xs.shape
      raise ArgumentError, "'mask' must be a boolean tensor." unless mask.dtype == :bool
    end

    # @parameter dim [Integer, Array(Integer), nil] the dimension(s) to reduce;
    #   nil reduces all of them.
    # @returns [Array(Integer)] the dimensions in the form torch.rb expects,
    #   where the empty array reduces all of them.
    def reduction_dims(dim)
      dim.nil? ? [] : Array(dim)
    end

    # Shared by the operators with a stable variant: the `stable` flag can be
    # set in the constructor and overridden on each call.
    module Stable
      # @attribute [Boolean] whether to use the stable variant by default.
      attr_accessor :stable

      # @parameter stable [Boolean] whether to use the stable variant by default.
      def initialize(stable: true)
        @stable = stable
      end

      def inspect
        "#{self.class.name.split("::").last}(stable: #{stable})"
      end

      private

      def stable?(override)
        override.nil? ? stable : override
      end
    end

    # Abstract base class of the connective operators.
    class ConnectiveOperator
      # @raises [NotImplementedError] always: subclasses implement #call.
      def call(*args, **kwargs)
        raise NotImplementedError, "#{self.class} must implement #call"
      end

      def inspect
        "#{self.class.name.split("::").last}()"
      end

      def to_s
        inspect
      end
    end

    # Abstract base class of the unary connective operators.
    class UnaryConnectiveOperator < ConnectiveOperator
    end

    # Abstract base class of the binary connective operators.
    class BinaryConnectiveOperator < ConnectiveOperator
    end

    # Standard negation: 1 - x.
    class NotStandard < UnaryConnectiveOperator
      def call(x)
        1.0 - x
      end
    end

    # Gödel negation: 1 if x == 0, 0 otherwise.
    class NotGodel < UnaryConnectiveOperator
      def call(x)
        Torch.eq(x, 0.0).float
      end
    end

    # Gödel conjunction: min(x, y).
    class AndMin < BinaryConnectiveOperator
      def call(x, y)
        Torch.minimum(x, y)
      end
    end

    # Goguen (product) conjunction: x * y.
    class AndProd < BinaryConnectiveOperator
      include Stable

      def call(x, y, stable: nil)
        x, y = FuzzyOps.pi_0(x), FuzzyOps.pi_0(y) if stable?(stable)
        Torch.mul(x, y)
      end
    end

    # Łukasiewicz conjunction: max(x + y - 1, 0).
    class AndLuk < BinaryConnectiveOperator
      def call(x, y)
        Torch.maximum(x + y - 1.0, Torch.zeros_like(x))
      end
    end

    # Gödel disjunction: max(x, y).
    class OrMax < BinaryConnectiveOperator
      def call(x, y)
        Torch.maximum(x, y)
      end
    end

    # Goguen disjunction (probabilistic sum): x + y - x * y.
    class OrProbSum < BinaryConnectiveOperator
      include Stable

      def call(x, y, stable: nil)
        x, y = FuzzyOps.pi_1(x), FuzzyOps.pi_1(y) if stable?(stable)
        x + y - Torch.mul(x, y)
      end
    end

    # Łukasiewicz disjunction: min(x + y, 1).
    class OrLuk < BinaryConnectiveOperator
      def call(x, y)
        Torch.minimum(x + y, Torch.ones_like(x))
      end
    end

    # Kleene-Dienes implication: max(1 - x, y).
    class ImpliesKleeneDienes < BinaryConnectiveOperator
      def call(x, y)
        Torch.maximum(1.0 - x, y)
      end
    end

    # Gödel implication: 1 if x <= y, y otherwise.
    class ImpliesGodel < BinaryConnectiveOperator
      def call(x, y)
        Torch.where(Torch.le(x, y), Torch.ones_like(x), y)
      end
    end

    # Reichenbach implication: 1 - x + x * y.
    class ImpliesReichenbach < BinaryConnectiveOperator
      include Stable

      def call(x, y, stable: nil)
        x, y = FuzzyOps.pi_0(x), FuzzyOps.pi_1(y) if stable?(stable)
        1.0 - x + Torch.mul(x, y)
      end
    end

    # Goguen implication: 1 if x <= y, y / x otherwise.
    class ImpliesGoguen < BinaryConnectiveOperator
      include Stable

      def call(x, y, stable: nil)
        x = FuzzyOps.pi_0(x) if stable?(stable)
        Torch.where(Torch.le(x, y), Torch.ones_like(x), Torch.div(y, x))
      end
    end

    # Łukasiewicz implication: min(1 - x + y, 1).
    class ImpliesLuk < BinaryConnectiveOperator
      def call(x, y)
        Torch.minimum(1.0 - x + y, Torch.ones_like(x))
      end
    end

    # Equivalence, as the conjunction of the implications in both directions.
    class Equiv < BinaryConnectiveOperator
      attr_reader :and_op, :implies_op

      # @parameter and_op [BinaryConnectiveOperator] the conjunction.
      # @parameter implies_op [BinaryConnectiveOperator] the implication.
      def initialize(and_op, implies_op)
        @and_op = and_op
        @implies_op = implies_op
      end

      def call(x, y)
        and_op.call(implies_op.call(x, y), implies_op.call(y, x))
      end

      def inspect
        "Equiv(and_op: #{and_op.inspect}, implies_op: #{implies_op.inspect})"
      end
    end

    # Abstract base class of the aggregation operators.
    #
    # Every aggregator responds to `call(xs, dim: nil, keepdim: false, mask: nil)`:
    # `dim` is the dimension (or dimensions) to aggregate, nil for all of them,
    # and `mask` is a boolean tensor shaped like `xs` that excludes the values
    # where it is false (used for guarded quantification).
    class AggregationOperator
      # @raises [NotImplementedError] always: subclasses implement #call.
      def call(*args, **kwargs)
        raise NotImplementedError, "#{self.class} must implement #call"
      end

      def inspect
        "#{self.class.name.split("::").last}()"
      end

      def to_s
        inspect
      end

      private

      # Mean of `xs` over `dims`, counting only the values where `mask` is true.
      def masked_mean(xs, mask, dims, keepdim)
        FuzzyOps.check_mask(mask, xs)
        numerator = Torch.sum(Torch.where(mask.logical_not, Torch.zeros_like(xs), xs), dim: dims, keepdim: keepdim)
        denominator = Torch.sum(mask, dim: dims, keepdim: keepdim)
        Torch.div(numerator, denominator)
      end
    end

    # Shared by the pMean aggregators: `p` and `stable` can be set in the
    # constructor and overridden on each call.
    module PMeanParameters
      # @attribute [Numeric] the exponent of the generalized mean.
      attr_accessor :p

      # @attribute [Boolean] whether to use the stable variant by default.
      attr_accessor :stable

      # @parameter p [Numeric] the exponent of the generalized mean.
      # @parameter stable [Boolean] whether to use the stable variant by default.
      def initialize(p: 2, stable: true)
        @p = p
        @stable = stable
      end

      def inspect
        "#{self.class.name.split("::").last}(p: #{p}, stable: #{stable})"
      end
    end

    # Minimum, an approximation of the universal quantifier.
    class AggregMin < AggregationOperator
      def call(xs, dim: nil, keepdim: false, mask: nil)
        unless mask.nil?
          FuzzyOps.check_mask(mask, xs)
          # 1 is the largest truth value, so masked-out values never win the
          # minimum. As in LTNtorch, this turns the result into float64.
          xs = Torch.where(mask.logical_not, 1.0, xs.double)
        end
        Torch.amin(xs, dim: FuzzyOps.reduction_dims(dim), keepdim: keepdim)
      end
    end

    # Arithmetic mean.
    class AggregMean < AggregationOperator
      def call(xs, dim: nil, keepdim: false, mask: nil)
        dims = FuzzyOps.reduction_dims(dim)
        return masked_mean(xs, mask, dims, keepdim) unless mask.nil?

        Torch.mean(xs, dim: dims, keepdim: keepdim)
      end
    end

    # Generalized mean (pMean), an approximation of the existential quantifier:
    # (mean(x ** p)) ** (1 / p).
    class AggregPMean < AggregationOperator
      include PMeanParameters

      def call(xs, dim: nil, keepdim: false, mask: nil, p: nil, stable: nil)
        p = self.p if p.nil?
        stable = self.stable if stable.nil?
        dims = FuzzyOps.reduction_dims(dim)

        xs = FuzzyOps.pi_0(xs) if stable
        xs = Torch.pow(xs, p)
        mean = mask.nil? ? Torch.mean(xs, dim: dims, keepdim: keepdim) : masked_mean(xs, mask, dims, keepdim)
        Torch.pow(mean, 1.0 / p)
      end
    end

    # Generalized mean of the errors (pMeanError), an approximation of the
    # universal quantifier: 1 - (mean((1 - x) ** p)) ** (1 / p).
    class AggregPMeanError < AggregationOperator
      include PMeanParameters

      def call(xs, dim: nil, keepdim: false, mask: nil, p: nil, stable: nil)
        p = self.p if p.nil?
        stable = self.stable if stable.nil?
        dims = FuzzyOps.reduction_dims(dim)

        xs = FuzzyOps.pi_1(xs) if stable
        xs = Torch.pow(1.0 - xs, p)
        mean = mask.nil? ? Torch.mean(xs, dim: dims, keepdim: keepdim) : masked_mean(xs, mask, dims, keepdim)
        1.0 - Torch.pow(mean, 1.0 / p)
      end
    end

    # Aggregates the truth values of the closed formulas of a knowledge base
    # into its overall satisfaction level, typically to build a loss as
    # `1 - sat_agg.(...)`.
    class SatAgg
      attr_reader :agg_op

      # @parameter agg_op [AggregationOperator] how to aggregate the truth values.
      # @raises [TypeError] if `agg_op` is not an aggregation operator.
      def initialize(agg_op = AggregPMeanError.new(p: 2))
        unless agg_op.is_a?(AggregationOperator)
          raise TypeError, "SatAgg.new: argument 'agg_op' (position 1) must be an AggregationOperator, not #{agg_op.class}"
        end

        @agg_op = agg_op
      end

      # @parameter closed_formulas [Array(LTNObject | Torch::Tensor)] scalar truth values.
      # @returns [Torch::Tensor] the aggregated (scalar) truth value.
      # @raises [TypeError] if a formula is neither an LTNObject nor a tensor.
      # @raises [ArgumentError] if a formula is not closed (its value is not a
      #   scalar) or not in [0, 1].
      def call(*closed_formulas)
        unless closed_formulas.all? { |f| f.is_a?(LTNObject) || f.is_a?(Torch::Tensor) }
          raise TypeError, "Expected parameter 'closed_formulas' to be LTNObjects and/or tensors, " \
            "but got #{closed_formulas.map(&:class).inspect}"
        end

        truth_values = closed_formulas.map { |f| f.is_a?(LTNObject) ? f.value : f }
        unless truth_values.all? { |v| v.shape == [] }
          raise ArgumentError, "Expected parameter 'closed_formulas' to be LTNObjects and/or tensors containing " \
            "scalars, but got the following shapes: #{truth_values.map(&:shape).inspect}"
        end

        truth_values = Torch.stack(truth_values, dim: 0)
        FuzzyOps.check_values(truth_values)
        agg_op.call(truth_values, dim: 0)
      end

      def inspect
        "SatAgg(agg_op: #{agg_op.inspect})"
      end

      def to_s
        inspect
      end
    end
  end
end

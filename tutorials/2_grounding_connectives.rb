#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's tutorials/2-grounding_connectives.ipynb, copyright 2021-2024 by Tommaso Carraro;
# see the notebook for the full discussion.
#
# Tutorial 2: grounding connectives and quantifiers, including diagonal and
# guarded quantification.
#
#   bundle exec ruby tutorials/2_grounding_connectives.rb

require "ltn"

Torch.manual_seed(2020)

def show(label, value)
  puts "#{label.ljust(36)} #{value.inspect}"
end

puts "== Connectives"
# Connectives are grounded with fuzzy semantics (LTN::FuzzyOps). The
# recommended ones are the standard negation 1 - u, the product t-norm u v,
# the probabilistic sum u + v - u v, and the Reichenbach implication
# 1 - u + u v (see tutorial 2b for why).
not_ = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
and_ = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
or_ = LTN::Connective.new(LTN::FuzzyOps::OrProbSum.new)
implies = LTN::Connective.new(LTN::FuzzyOps::ImpliesReichenbach.new)

x = LTN::Variable.new("x", Torch.randn(10, 2))
y = LTN::Variable.new("y", Torch.randn(5, 2))
c1 = LTN::Constant.new(Torch.tensor([0.5, 0.0]))
c2 = LTN::Constant.new(Torch.tensor([4.0, 2.0]))
# A predicate measuring the similarity of two points.
eq = LTN::Predicate.new { |a, b| Torch.exp(Torch.neg(Torch.norm(a - b, 2, dim: 1))) }

show("Eq(c1, c2)", eq.call(c1, c2).value)
show("not Eq(c1, c2)", not_.call(eq.call(c1, c2)).value)
show("Eq(c1, c2) -> Eq(c2, c1)", implies.call(eq.call(c1, c2), eq.call(c2, c1)).value)
# LTN::Connective broadcasts its operands: a formula has one axis per
# variable appearing in any of its sub-formulas.
show("(Eq(x, c1) and Eq(x, c2)).shape", and_.call(eq.call(x, c1), eq.call(x, c2)).shape)
show("(Eq(x, c1) or Eq(x, y)).shape", or_.call(eq.call(x, c1), eq.call(x, y)).shape)

puts "\n== Quantifiers"
# Quantifiers are grounded by aggregation operators. The recommended ones
# are the generalized mean pMean for exists, a soft maximum, and the
# generalized mean of the errors pMeanError for forall, a soft minimum.
forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists)

# Quantifying a variable aggregates its axis away; give several variables as
# an array.
show("Eq(x, y).shape", eq.call(x, y).shape)
show("(forall x: Eq(x, y)).shape", forall.call(x, eq.call(x, y)).shape)
show("forall x, y: Eq(x, y)", forall.call([x, y], eq.call(x, y)).value)
show("exists x, y: Eq(x, y)", exists.call([x, y], eq.call(x, y)).value)
show("forall x: exists y: Eq(x, y)", forall.call(x, exists.call(y, eq.call(x, y))).value)

# p sets how strict the aggregation is: as p grows, pMeanError tends to the
# minimum (forall), and pMean to the maximum (exists). It can be given per
# call.
show("forall x: Eq(x, c1), p = 2", forall.call(x, eq.call(x, c1), p: 2).value)
show("forall x: Eq(x, c1), p = 10", forall.call(x, eq.call(x, c1), p: 10).value)
show("exists x: Eq(x, c1), p = 2", exists.call(x, eq.call(x, c1), p: 2).value)
show("exists x: Eq(x, c1), p = 10", exists.call(x, eq.call(x, c1), p: 10).value)

puts "\n== Diagonal quantification"
# Sometimes a formula should only be evaluated on the i-th individuals of
# several variables together (e.g. samples and their labels), not on every
# combination of them. LTN.diag marks variables (with the same number of
# individuals) to be paired this way.
samples = Torch.randn(100, 2, 2) # 100 values in R^{2x2}
labels = Torch.randint(0, 3, [100], dtype: :int64) # one of 3 classes each
x = LTN::Variable.new("x", samples)
l = LTN::Variable.new("l", Torch.eye(3)[labels]) # one-hot labels

class ModelC < Torch::NN::Module
  def initialize
    super
    @elu = Torch::NN::ELU.new
    @softmax = Torch::NN::Softmax.new(dim: 1)
    @dense1 = Torch::NN::Linear.new(4, 5)
    @dense2 = Torch::NN::Linear.new(5, 3)
  end

  def forward(x, l)
    x = Torch.flatten(x, start_dim: 1)
    x = @softmax.call(@dense2.call(@elu.call(@dense1.call(x))))
    Torch.sum(x * l, dim: 1)
  end
end
c = LTN::Predicate.new(ModelC.new)

show("C(x, l).shape", c.call(x, l).shape) # all 100 x 100 combinations
LTN.diag(x, l)
show("C(x, l).shape after diag", c.call(x, l).shape) # the 100 pairs only
show("x.free_vars, l.free_vars", [x.free_vars, l.free_vars])
LTN.undiag(x, l)
show("C(x, l).shape after undiag", c.call(x, l).shape)

# In practice, use LTN.diag inside a quantifier: the quantifier resets the
# variables (LTN.undiag) afterwards. Call it before building the formula, as
# below, since it changes the variables' labels.
show("forall diag(x, l): C(x, l)", forall.call(LTN.diag(x, l), c.call(x, l)).value)
show("x.free_vars, l.free_vars", [x.free_vars, l.free_vars])

puts "\n== Guarded quantification"
# A guard restricts a quantifier to the individuals satisfying a boolean
# condition, which may involve other variables too: "there is a distance d
# such that all pairs of points closer than d are similar".
points = Torch.rand(50, 2)
x = LTN::Variable.new("x", points)
y = LTN::Variable.new("y", points)
d = LTN::Variable.new("d", Torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
dist = ->(a, b) { Torch.norm(a.value - b.value, 2, dim: 1).unsqueeze(1) }

result = exists.call(d,
  forall.call([x, y], eq.call(x, y), cond_vars: [x, y, d]) { |a, b, dd| dist.call(a, b).lt(dd.value) })
show("exists d: forall x, y with dist < d: Eq(x, y)", result.value)
# The guard is a function (or block) of the condition variables, returning a
# boolean tensor; it receives LTN objects, so it reads their values. Since
# only the pairs satisfying it are aggregated, gradients also flow only
# through them.

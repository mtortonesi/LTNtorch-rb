#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's tutorials/2b-operators-and-gradients.ipynb, copyright 2021-2024 by Tommaso Carraro;
# see the notebook, and the LTN paper (https://arxiv.org/abs/2012.13635), for the full discussion.
#
# Tutorial 2b: choosing fuzzy operators, and why their gradients matter.
#
#   bundle exec ruby tutorials/2b_operators_and_gradients.rb
#
# LTN learns by gradient descent on the truth values of formulas, so the
# choice of fuzzy semantics is not only about logic: it decides which
# truth values receive gradients, and how large they are.

require "ltn"

def show(label, value)
  puts "#{label.ljust(44)} #{value.inspect}"
end

puts "== Operators"
# Operators work directly on tensors of truth values.
x1 = Torch.tensor(0.4)
x2 = Torch.tensor(0.7)
and_prod = LTN::FuzzyOps::AndProd.new(stable: false) # `stable` is explained below
and_luk = LTN::FuzzyOps::AndLuk.new
show("product: 0.4 and 0.7", and_prod.call(x1, x2))
show("Lukasiewicz: 0.4 and 0.7", and_luk.call(x1, x2))

xs = Torch.tensor([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2, 0.1])
forall_min = LTN::FuzzyOps::AggregMin.new
forall_pme = LTN::FuzzyOps::AggregPMeanError.new(p: 4, stable: false)
show("min", forall_min.call(xs, dim: 0))
show("pMeanError, p = 4", forall_pme.call(xs, dim: 0))

puts "\n== Gradients"
# Lukasiewicz's conjunction has zero gradient whenever x1 + x2 <= 1: no
# learning signal at all for such operands.
x1 = Torch.tensor(0.3).requires_grad!(true)
x2 = Torch.tensor(0.5).requires_grad!(true)
y = and_luk.call(x1, x2)
y.backward
show("Lukasiewicz: 0.3 and 0.5, gradients", [y.item, x1.grad.item, x2.grad.item])

# The minimum only passes gradients to the smallest value: one example per
# step gets a learning signal.
xs = Torch.tensor([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2, 0.1]).requires_grad!(true)
y = forall_min.call(xs, dim: 0)
y.backward
show("min, gradients", [y.item, xs.grad.to_a])

# pMeanError passes gradients to every value, more to the ones further from
# the truth. But with all values at 1, its gradient is 0/0: NaN.
xs = Torch.tensor([1.0, 1.0, 1.0]).requires_grad!(true)
y = forall_pme.call(xs, dim: 0, p: 4)
y.backward
show("pMeanError of 1s, gradients", [y.item, xs.grad.to_a])

puts "\n== Stable operators"
# The stable variants (the default) first project the truth values slightly
# into (0, 1] or [0, 1) (LTN::FuzzyOps.pi_0 and pi_1, with epsilon 1e-4),
# keeping their results and gradients finite.
xs = Torch.tensor([1.0, 1.0, 1.0]).requires_grad!(true)
y = forall_pme.call(xs, dim: 0, p: 4, stable: true)
y.backward
show("stable pMeanError of 1s, gradients", [y.item, xs.grad.to_a])

# p trades off between a mean (p = 1, every value counts equally) and the
# minimum (large p, the worst values dominate): larger p focuses the
# gradients on the outliers.
[4, 20].each do |p|
  xs = Torch.tensor([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2, 0.1]).requires_grad!(true)
  y = forall_pme.call(xs, dim: 0, p: p)
  y.backward
  show("pMeanError, p = #{p}, gradients", [y.item.round(4), xs.grad.to_a.map { |g| g.round(4) }])
end

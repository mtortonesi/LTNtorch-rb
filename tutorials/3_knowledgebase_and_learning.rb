#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's tutorials/3-knowledgebase-and-learning.ipynb, copyright 2021-2024 by Tommaso Carraro;
# see the notebook for the full discussion.
#
# Tutorial 3: learning from a knowledge base.
#
#   bundle exec ruby tutorials/3_knowledgebase_and_learning.rb
#
# The learnable parameters of LTN are those of the groundings of its
# symbols. Training maximizes the satisfaction of the knowledge base, i.e.
# the aggregated truth value of its axioms (here with SatAgg), by gradient
# descent on the loss 1 - satisfaction.
#
# The task: two classes A and B of points in [0, 4]^2, of which only one
# example each is known, a and b. The knowledge base states that a is an A,
# b is a B, and that similar points belong to the same classes:
#
#   C(a, l_a)
#   C(b, l_b)
#   forall x1, x2, l: Sim(x1, x2) -> (C(x1, l) <-> C(x2, l))

require "ltn"

EPOCHS = Integer(ENV.fetch("EPOCHS", 2000))
BATCH_EPOCHS = Integer(ENV.fetch("BATCH_EPOCHS", 100))

Torch.manual_seed(2020)

# C(x, l): the probability, from a softmax over the two classes, that point x
# has class l (one-hot).
class ModelC < Torch::NN::Module
  def initialize
    super
    @elu = Torch::NN::ELU.new
    @softmax = Torch::NN::Softmax.new(dim: 1)
    @dense1 = Torch::NN::Linear.new(2, 5)
    @dense2 = Torch::NN::Linear.new(5, 5)
    @dense3 = Torch::NN::Linear.new(5, 2)
  end

  def forward(x, l)
    x = @elu.call(@dense1.call(x))
    x = @elu.call(@dense2.call(x))
    Torch.sum(@softmax.call(@dense3.call(x)) * l, dim: 1)
  end
end

points = Torch.tensor([[0.4, 0.3], [1.2, 0.3], [2.2, 1.3], [1.7, 1.0], [0.5, 0.5], [0.3, 1.5], [1.3, 1.1],
  [0.9, 1.7], [3.4, 3.3], [3.2, 3.3], [3.2, 2.3], [2.7, 2.0], [3.5, 3.5], [3.3, 2.5], [3.3, 1.1], [1.9, 3.7],
  [1.3, 3.5], [3.3, 1.1], [3.9, 3.7]])

c = LTN::Predicate.new(ModelC.new)
# Sim(u, v) = exp(-||u - v||): a similarity, not learned.
sim = LTN::Predicate.new { |u, v| Torch.exp(Torch.neg(Torch.sqrt(Torch.sum(Torch.square(u - v), dim: 1)))) }

x1 = LTN::Variable.new("x1", points)
x2 = LTN::Variable.new("x2", points)
a = LTN::Constant.new(Torch.tensor([3.3, 2.5]))
b = LTN::Constant.new(Torch.tensor([1.3, 1.1]))
l_a = LTN::Constant.new(Torch.tensor([1, 0]))
l_b = LTN::Constant.new(Torch.tensor([0, 1]))
l = LTN::Variable.new("l", Torch.tensor([[1, 0], [0, 1]]))

implies = LTN::Connective.new(LTN::FuzzyOps::ImpliesReichenbach.new)
equiv = LTN::Connective.new(LTN::FuzzyOps::Equiv.new(LTN::FuzzyOps::AndProd.new, LTN::FuzzyOps::ImpliesReichenbach.new))
forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
sat_agg = LTN::FuzzyOps::SatAgg.new # pMeanError by default

puts "== Learning"
optimizer = Torch::Optim::Adam.new(c.parameters, lr: 0.001)
loss = nil
EPOCHS.times do |epoch|
  optimizer.zero_grad
  loss = 1.0 - sat_agg.call(
    c.call(a, l_a),
    c.call(b, l_b),
    forall.call([x1, x2, l], implies.call(sim.call(x1, x2), equiv.call(c.call(x1, l), c.call(x2, l))))
  )
  loss.backward
  optimizer.step
  puts format("Epoch %d: Sat Level %.3f", epoch, 1 - loss.item) if (epoch % 200).zero?
end
puts format("Training finished with Sat Level %.3f", 1 - loss.item)

puts "\nC(x, l_a) for each point (a = [3.3, 2.5], b = [1.3, 1.1]):"
Torch.no_grad do
  c.call(x1, l_a).value.to_a.each_with_index do |truth, i|
    puts format("  [%.1f, %.1f]  %.2f", *points[i].to_a, truth)
  end
end

puts "\n== Learning in mini-batches"
# With many individuals, a variable can be grounded with a mini-batch of
# them at each step, like any other training data. Here: 10,000 points of
# [0, 4]^2, with a = [3, 3] and b = [1, 1] among them.
points = Torch.rand(10_000, 2) * 4
points[-1] = Torch.tensor([3.0, 3.0])
points[-2] = Torch.tensor([1.0, 1.0])
a = LTN::Constant.new(Torch.tensor([3.0, 3.0]))
b = LTN::Constant.new(Torch.tensor([1.0, 1.0]))

c = LTN::Predicate.new(ModelC.new) # a new, untrained C
optimizer = Torch::Optim::Adam.new(c.parameters, lr: 0.001)
BATCH_EPOCHS.times do |epoch|
  Torch.randperm(points.shape[0]).split(512).each do |indices|
    batch = points[indices]
    x1 = LTN::Variable.new("x1", batch)
    x2 = LTN::Variable.new("x2", batch)
    optimizer.zero_grad
    loss = 1.0 - sat_agg.call(
      c.call(a, l_a),
      c.call(b, l_b),
      forall.call([x1, x2, l], implies.call(sim.call(x1, x2), equiv.call(c.call(x1, l), c.call(x2, l))))
    )
    loss.backward
    optimizer.step
  end
  puts format("Epoch %d: Sat Level %.3f", epoch, 1 - loss.item) if (epoch % 10).zero?
end
puts format("Training finished with Sat Level %.3f", 1 - loss.item)

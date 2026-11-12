#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's tutorials/1-grounding_non_logical_symbols.ipynb, copyright 2021-2024 by
# Tommaso Carraro; see the notebook for the full discussion.
#
# Tutorial 1: grounding the non-logical symbols (constants, predicates,
# functions, variables).
#
#   bundle exec ruby tutorials/1_grounding_non_logical_symbols.rb
#
# In LTN's Real Logic, a *grounding* maps every term of a first-order
# language to a tensor of real numbers, and every formula to a truth value in
# [0, 1]. This tutorial covers the signature: constants, predicates,
# functions and variables. Tutorial 2 covers connectives and quantifiers.

require "ltn"

Torch.manual_seed(2020)

def show(label, value)
  puts "#{label.ljust(24)} #{value.inspect}"
end

puts "== Constants"
# A constant is an individual of the domain, grounded as a tensor of any
# rank: here a vector of R^2 and a 2x3 matrix.
c1 = LTN::Constant.new(Torch.tensor([2.1, 3.0]))
c2 = LTN::Constant.new(Torch.tensor([[4.2, 3.0, 2.5], [4.0, -1.3, 1.8]]))
# A trainable constant is an embedding: its tensor requires grad, and is
# learned with the rest of the knowledge base (see example 7).
c3 = LTN::Constant.new(Torch.tensor([0.0, 0.0]), trainable: true)
show("c1.value", c1.value)
show("c2.shape", c2.shape)
show("c3.value", c3.value)
# Constants are LTNObjects: a value, and the labels of the free variables
# appearing in it (none, for a constant).
show("c1.free_vars", c1.free_vars)

puts "\n== Predicates"
# A predicate maps individuals to truth values in [0, 1]. It is grounded
# either by a function (or block), for fixed predicates without trainable
# weights, or by a Torch::NN::Module, for learnable ones.
mu = LTN::Constant.new(Torch.tensor([2.0, 3.0]))
p1 = LTN::Predicate.new { |x| Torch.exp(Torch.neg(Torch.norm(x - mu.value, 2, dim: 1))) }

class ModelP2 < Torch::NN::Module
  def initialize
    super
    @elu = Torch::NN::ELU.new
    @sigmoid = Torch::NN::Sigmoid.new
    @dense1 = Torch::NN::Linear.new(2, 5)
    @dense2 = Torch::NN::Linear.new(5, 1) # one value in [0, 1]
  end

  def forward(x)
    @sigmoid.call(@dense2.call(@elu.call(@dense1.call(x))))
  end
end
p2 = LTN::Predicate.new(ModelP2.new)

# Predicates take LTN objects, and return an LTNObject whose value holds the
# truth values.
show("P1(c1)", p1.call(c1).value)
show("P1(LTN::Constant [4.5, 0.8])", p1.call(LTN::Constant.new(Torch.tensor([4.5, 0.8]))).value)
show("P2(LTN::Constant [3, 4.8])", p2.call(LTN::Constant.new(Torch.tensor([3.0, 4.8]))).value)
show("P1(c1) as an LTNObject", p1.call(c1))

# LTN adds a batch dimension in front of the inputs it gives to the model, so
# models work on dim 1, the features. A binary predicate gets two tensors.
class ModelP4 < Torch::NN::Module
  def initialize
    super
    @elu = Torch::NN::ELU.new
    @sigmoid = Torch::NN::Sigmoid.new
    @dense1 = Torch::NN::Linear.new(4, 5)
    @dense2 = Torch::NN::Linear.new(5, 1)
  end

  def forward(x, y)
    x = Torch.cat([x, y], dim: 1)
    @sigmoid.call(@dense2.call(@elu.call(@dense1.call(x))))
  end
end
p4 = LTN::Predicate.new(ModelP4.new)
c4 = LTN::Constant.new(Torch.tensor([4.5, 0.8]))
show("P4(c1, c4)", p4.call(c1, c4).value)

puts "\n== Functions"
# A function maps individuals to an individual (a tensor of any shape), and
# is grounded like a predicate: by a function or by a module.
f1 = LTN::Function.new { |x, y| x - y }

class MyModel < Torch::NN::Module
  def initialize
    super
    @dense1 = Torch::NN::Linear.new(2, 10)
    @dense2 = Torch::NN::Linear.new(10, 5)
    @relu = Torch::NN::ReLU.new
  end

  def forward(x)
    @dense2.call(@relu.call(@dense1.call(x)))
  end
end
f2 = LTN::Function.new(MyModel.new)
show("f1(c1, c4)", f1.call(c1, c4).value)
show("f2(c1)", f2.call(c1).value)

puts "\n== Variables"
# A variable is a labeled sequence of individuals (not a set: an individual
# may occur twice). Its free_vars are its label.
x = LTN::Variable.new("x", Torch.randn(10, 2))
y = LTN::Variable.new("y", Torch.randn(5, 2))
show("x.free_vars", x.free_vars)

# Evaluating a symbol on variables evaluates it on every combination of
# their individuals: one axis per variable, in the order in which they
# appear, and free_vars tells which axis is which.
res1 = p4.call(x, y)
show("P4(x, y).shape", res1.shape)
show("P4(x, y).free_vars", res1.free_vars)
show("P4(x_3, y_1)", res1.value[2, 0])

# A function's outputs keep their own dimensions after the variables' axes:
# f1 maps to R^2, so f1(x, y) is 10 x 5 x 2.
res2 = f1.call(x, y)
show("f1(x, y).shape", res2.shape)
show("f1(x, y).free_vars", res2.free_vars)

# Constants add no axis.
res3 = p4.call(c1, y)
show("P4(c1, y).shape", res3.shape)
show("P4(c1, y).free_vars", res3.free_vars)

puts "\n== Variables made of trainable constants"
# A variable can be built from the values of trainable constants, so that
# gradients flow back to them through any formula using the variable.
c1 = LTN::Constant.new(Torch.tensor([2.1, 3.0]), trainable: true)
c2 = LTN::Constant.new(Torch.tensor([4.5, 0.8]), trainable: true)
x = LTN::Variable.new("x", Torch.stack([c1.value, c2.value]))
p2.call(x).value.sum.backward
show("gradient of c1", c1.value.grad)
show("gradient of c2", c2.value.grad)

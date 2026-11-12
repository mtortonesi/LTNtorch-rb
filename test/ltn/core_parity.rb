# frozen_string_literal: true

# Checks LTN's core (broadcasting, predicates, functions, connectives,
# quantifiers, guarded and diagonal quantification, and the gradients of a
# small knowledge base) against reference results from LTNtorch, generated
# by test/fixtures/core.py. Each scenario below mirrors the one with the same
# name in that script.

require "ltn"
require_relative "../support/fixtures"
require_relative "../support/torch_test_helpers"

CORE_FIXTURE = Fixtures.load("core")

module CoreParityHelpers
  # Groundings shared with test/fixtures/core.py.
  P1 = ->(a) { Torch.sigmoid(Torch.sum(a, dim: 1)) }
  P2 = ->(a, b) { Torch.sigmoid(Torch.sum(Torch.cat([a, b], dim: 1), dim: 1)) }
  P2_FLAT = ->(a, b) { Torch.sigmoid(Torch.sum(Torch.cat([a, Torch.flatten(b, start_dim: 1)], dim: 1), dim: 1)) }
  EQ = ->(a, b) { Torch.exp(Torch.neg(Torch.norm(a - b, 2, dim: 1))) }
  F2 = ->(a, b) { Torch.cat([a, b], dim: 1) }
  F_MEAN = ->(a) { Torch.mean(a, dim: 1) }
  F_FLAT = ->(a, b) { Torch.cat([a, Torch.flatten(b, start_dim: 1)], dim: 1) }

  def data(name) = Torch.tensor(CORE_FIXTURE["data"].fetch(name))

  def x = LTN::Variable.new("x", data("x"))

  def y = LTN::Variable.new("y", data("y"))

  def z = LTN::Variable.new("z", data("z"))

  def c = LTN::Constant.new(data("c"))

  def xy = LTN::LTNObject.new(data("t2"), ["x", "y"])

  def yz = LTN::LTNObject.new(data("t3"), ["y", "z"])

  def forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)

  def exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists)

  def and_prod = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)

  def implies = LTN::Connective.new(LTN::FuzzyOps::ImpliesReichenbach.new)

  def not_ = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)

  def predicate(func) = LTN::Predicate.new(func: func)

  def function(func) = LTN::Function.new(func: func)

  def expect_object(object, expected)
    expect(object.free_vars).to be == expected["free_vars"]
    expect_tensor(object.value, expected["value"])
  end

  def expect_tensor(tensor, expected)
    matches = Fixtures.matches?(tensor, expected)
    inform("got #{Fixtures.describe(tensor)}, expected #{expected.inspect}") unless matches
    expect(matches).to be_truthy
  end
end

describe "LTN core parity with LTNtorch" do
  include CoreParityHelpers

  it "was checked against reference values from the same PyTorch version as its LibTorch" do
    expect(CORE_FIXTURE["torch_version"]).to be == "2.15.0"
  end

  with "LTN.process_ltn_objects" do
    {
      "const_const" => -> { [c, LTN::Constant.new(data("x"))] },
      "const_var" => -> { [c, x] },
      "var_var" => -> { [x, y] },
      "three_vars" => -> { [y, x, z] },
      "shared_var" => -> { [xy, yz] },
      "extra_dims" => -> {
        [LTN::LTNObject.new(Torch.arange(2 * 4 * 3, dtype: :float32).view(2, 4, 3), ["x", "y"]), z]
      },
      "diag" => -> { LTN.diag(LTN::Variable.new("x", data("x")), LTN::Variable.new("w", data("w"))) }
    }.each do |name, objects|
      it "matches for #{name}" do
        expected = CORE_FIXTURE["process"].fetch(name)
        processed, vars, n_individuals = LTN.process_ltn_objects(instance_exec(&objects))

        expect(vars).to be == expected["vars"]
        expect(n_individuals).to be == expected["n_individuals"]
        expect(processed.size).to be == expected["objects"].size
        processed.zip(expected["objects"]).each { |object, reference| expect_object(object, reference) }
      end
    end
  end

  with "predicates and functions" do
    {
      "p1_x" => -> { predicate(CoreParityHelpers::P1).call(x) },
      "p1_z" => -> { predicate(CoreParityHelpers::P1).call(z) },
      "p2_x_y" => -> { predicate(CoreParityHelpers::P2).call(x, y) },
      "p2_y_x" => -> { predicate(CoreParityHelpers::P2).call(y, x) },
      "p2_x_c" => -> { predicate(CoreParityHelpers::P2_FLAT).call(x, c) },
      "p1_c" => -> { predicate(->(a) { Torch.sigmoid(Torch.sum(a)) }).call(c) },
      "f2_x_y" => -> { function(CoreParityHelpers::F2).call(x, y) },
      "f_mean_x" => -> { function(CoreParityHelpers::F_MEAN).call(x) },
      "f_flat_x_c" => -> { function(CoreParityHelpers::F_FLAT).call(x, c) },
      "p2_f2" => -> { predicate(CoreParityHelpers::P1).call(function(CoreParityHelpers::F2).call(x, y)) }
    }.each do |name, scenario|
      it "matches for #{name}" do
        expect_object(instance_exec(&scenario), CORE_FIXTURE["predicates"].fetch(name))
      end
    end
  end

  with "connectives" do
    {
      "and_xy_yz" => -> { and_prod.call(xy, yz) },
      "and_yz_xy" => -> { and_prod.call(yz, xy) },
      "implies_p1x_p2xy" => -> { implies.call(predicate(CoreParityHelpers::P1).call(x), predicate(CoreParityHelpers::P2).call(x, y)) },
      "not_p1z" => -> { not_.call(predicate(CoreParityHelpers::P1).call(z)) },
      "nested" => -> {
        and_prod.call(implies.call(predicate(CoreParityHelpers::P1).call(x), predicate(CoreParityHelpers::P2).call(x, y)), not_.call(predicate(CoreParityHelpers::P1).call(z)))
      }
    }.each do |name, scenario|
      it "matches for #{name}" do
        expect_object(instance_exec(&scenario), CORE_FIXTURE["connectives"].fetch(name))
      end
    end
  end

  with "quantifiers" do
    sum_x_gt = ->(a) { Torch.sum(a.value, dim: 1).gt(1.0) }
    sum_xy = ->(a, b) { Torch.logical_and(Torch.sum(a.value, dim: 1).gt(1.0), Torch.sum(b.value, dim: 1).gt(0.0)) }
    near = ->(a, b) { Torch.abs(Torch.sum(a.value, dim: 1) - Torch.sum(b.value, dim: 1)).lt(0.5) }
    dist = ->(a, b) { Torch.norm(a.value - b.value, 2, dim: 1).unsqueeze(1) }
    within_d = ->(a, b, d) { dist.call(a, b).lt(d.value) }

    {
      "forall_x" => -> { forall.call(x, predicate(CoreParityHelpers::P2).call(x, y)) },
      "forall_y" => -> { forall.call(y, predicate(CoreParityHelpers::P2).call(x, y)) },
      "forall_xy" => -> { forall.call([x, y], predicate(CoreParityHelpers::P2).call(x, y)) },
      "forall_x_exists_y" => -> {
        x_, y_ = x, y
        forall.call(x_, exists.call(y_, predicate(CoreParityHelpers::P2).call(x_, y_)))
      },
      "exists_y_forall_x" => -> {
        x_, y_ = x, y
        exists.call(y_, forall.call(x_, predicate(CoreParityHelpers::P2).call(x_, y_)))
      },
      "min_mean" => -> {
        LTN::Quantifier.new(LTN::FuzzyOps::AggregMin.new, :forall)
          .call(y, LTN::Quantifier.new(LTN::FuzzyOps::AggregMean.new, :exists).call(z, yz))
      },
      "forall_y_of_xy_yz" => -> { forall.call(y, and_prod.call(xy, yz)) },
      "kwargs_p" => -> { forall.call(x, predicate(CoreParityHelpers::P2).call(x, y), p: 4) },
      "guard_x" => -> {
        x_ = x
        forall.call(x_, predicate(CoreParityHelpers::P2).call(x_, y), cond_vars: x_, cond_fn: sum_x_gt)
      },
      "guard_xy" => -> {
        x_, y_ = x, y
        forall.call(x_, predicate(CoreParityHelpers::P2).call(x_, y_), cond_vars: [x_, y_], cond_fn: sum_xy)
      },
      "guard_xy_exists" => -> {
        x_, y_ = x, y
        exists.call(x_, predicate(CoreParityHelpers::P2).call(x_, y_), cond_vars: [x_, y_], cond_fn: sum_xy)
      },
      "guard_both" => -> {
        x_, y_ = x, y
        forall.call([x_, y_], predicate(CoreParityHelpers::P2).call(x_, y_), cond_vars: [x_, y_], cond_fn: sum_xy)
      },
      "guard_x_over_xy" => -> {
        x_, y_ = x, y
        forall.call([x_, y_], predicate(CoreParityHelpers::P2).call(x_, y_), cond_vars: x_, cond_fn: sum_x_gt)
      },
      "guard_y_over_x" => -> {
        x_, y_ = x, y
        forall.call(y_, predicate(CoreParityHelpers::P2).call(x_, y_), cond_vars: x_, cond_fn: sum_x_gt)
      },
      "guard_not_in_formula" => -> {
        x_, y_ = x, y
        forall.call(x_, predicate(CoreParityHelpers::P1).call(x_), cond_vars: [x_, y_], cond_fn: sum_xy)
      },
      "diag" => -> {
        v1, v2 = LTN::Variable.new("v1", data("x")), LTN::Variable.new("v2", data("w"))
        forall.call(LTN.diag(v1, v2), predicate(CoreParityHelpers::P2).call(v1, v2))
      },
      "diag_guard" => -> {
        v1, v2 = LTN::Variable.new("v1", data("x")), LTN::Variable.new("v2", data("w"))
        forall.call(LTN.diag(v1, v2), predicate(CoreParityHelpers::P2).call(v1, v2), cond_vars: [v1, v2], cond_fn: near)
      },
      "eq_d" => -> {
        xp, yp, d = LTN::Variable.new("x", data("points")), LTN::Variable.new("y", data("points")),
          LTN::Variable.new("d", data("d"))
        exists.call(d, forall.call([xp, yp], predicate(CoreParityHelpers::EQ).call(xp, yp), cond_vars: [xp, yp, d], cond_fn: within_d))
      },
      "eq_d_diag" => -> {
        xp, yp, d = LTN::Variable.new("x", data("points")), LTN::Variable.new("y", data("points2")),
          LTN::Variable.new("d", data("d"))
        exists.call(d,
          forall.call(LTN.diag(xp, yp), predicate(CoreParityHelpers::EQ).call(xp, yp), cond_vars: [xp, yp, d], cond_fn: within_d))
      }
    }.each do |name, scenario|
      it "matches for #{name}" do
        expect_object(instance_exec(&scenario), CORE_FIXTURE["quantifiers"].fetch(name))
      end
    end

    it "resets diagonal quantification afterwards" do
      v1, v2 = LTN::Variable.new("v1", data("x")), LTN::Variable.new("v2", data("w"))
      forall.call(LTN.diag(v1, v2), predicate(CoreParityHelpers::P2).call(v1, v2))
      expected = CORE_FIXTURE["quantifiers"]["diag_reset"]

      expect(v1.free_vars).to be == expected["v1"]
      expect(v2.free_vars).to be == expected["v2"]
    end
  end

  with "LTN::Quantifier.compute_mask" do
    sum_lt = ->(a) { Torch.sum(a.value, dim: 1).lt(0.3) }

    {
      "y_over_x" => -> { [[y], sum_lt, ["x"]] },
      "xy_over_xy" => -> { [[x, y], ->(a, b) { Torch.sum(a.value, dim: 1).lt(Torch.sum(b.value, dim: 1)) }, ["x", "y"]] },
      "y_over_xy" => -> { [[y], sum_lt, ["x", "y"]] },
      "z_not_in_formula" => -> { [[z], ->(a) { Torch.sum(a.value, dim: 1).gt(0.0) }, ["x"]] }
    }.each do |name, arguments|
      it "matches for #{name}" do
        cond_vars, cond_fn, aggregation_vars = instance_exec(&arguments)
        formula, mask = LTN::Quantifier.compute_mask(xy, cond_vars, cond_fn, aggregation_vars)
        expected = CORE_FIXTURE["masks"].fetch(name)

        expect_object(formula, expected["formula"])
        expect_object(mask, expected["mask"])
      end
    end
  end

  with "a knowledge base" do
    it "has the same axioms, satisfaction, loss and gradients as in LTNtorch" do
      expected = CORE_FIXTURE["knowledge_base"]
      model = Torch::NN::Linear.new(5, 1)
      Torch.no_grad do
        model.weight.copy!(data("linear_weight"))
        model.bias.copy!(data("linear_bias"))
      end
      a = predicate(->(u, v) { Torch.sigmoid(model.call(Torch.cat([u, v], dim: 1))).squeeze(1) })
      k = LTN::Constant.new(data("k"), trainable: true)
      x_, y_ = x, y

      axioms = [
        forall.call([x_, y_], a.call(x_, y_)),
        forall.call(x_, a.call(x_, k)),
        forall.call(x_, exists.call(y_, implies.call(predicate(CoreParityHelpers::P1).call(x_), a.call(x_, y_)))),
        forall.call(x_, not_.call(a.call(x_, k)), cond_vars: x_) { |u| Torch.sum(u.value, dim: 1).gt(0.0) }
      ]
      sat = LTN::FuzzyOps::SatAgg.new.call(*axioms)
      loss = 1.0 - sat
      loss.backward

      axioms.zip(expected["axioms"]).each { |axiom, reference| expect_tensor(axiom.value, reference) }
      expect_tensor(sat, expected["sat"])
      expect_tensor(loss, expected["loss"])
      expect_tensor(model.weight.grad, expected["grad_weight"])
      expect_tensor(model.bias.grad, expected["grad_bias"])
      expect_tensor(k.value.grad, expected["grad_k"])
    end
  end
end

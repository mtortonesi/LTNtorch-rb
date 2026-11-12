# frozen_string_literal: true

require "ltn"

describe LTN::Quantifier do
  let(:x) { LTN::Variable.new("x", Torch.tensor([[0.9, 0.4, 0.2], [-0.5, 0.3, 0.1]])) }
  let(:y) { LTN::Variable.new("y", Torch.tensor([[0.3, 0.2], [-0.1, -0.1], [-0.6, 0.2], [0.5, 0.3]])) }
  let(:c) { LTN::Constant.new(Torch.tensor([1, 2, 3, 4])) }
  let(:p) { LTN::Predicate.new { |a, b| Torch.sigmoid(Torch.sum(Torch.cat([a, b], dim: 1), dim: 1)) } }
  let(:pmean_error) { LTN::FuzzyOps::AggregPMeanError.new(p: 2) }
  let(:forall) { subject.new(pmean_error, :forall) }
  let(:exists) { subject.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists) }
  let(:sum_x_gt) { ->(a) { Torch.sum(a.value, dim: 1).gt(1.0) } }

  with "construction" do
    it "wants an aggregation operator" do
      expect { subject.new(LTN::FuzzyOps::OrLuk.new, :exists) }.to raise_exception(TypeError, message: be =~ /AggregationOperator/)
    end

    it "accepts :forall and :exists, and LTNtorch's 'f' and 'e'" do
      expect(subject.new(pmean_error, "f").quantifier).to be == :forall
      expect(subject.new(pmean_error, "e").quantifier).to be == :exists
      expect { subject.new(pmean_error, "a") }.to raise_exception(ArgumentError)
      expect { subject.new(pmean_error, :some) }.to raise_exception(ArgumentError)
    end

    it "describes itself" do
      expect(exists.inspect).to be == "Quantifier(agg_op=AggregPMean(p: 2, stable: true), quantifier=:exists)"
      expect(forall.to_s).to be == "Quantifier(agg_op=AggregPMeanError(p: 2, stable: true), quantifier=:forall)"
      expect(forall.agg_op).to be_equal(pmean_error)
    end
  end

  with "argument checks" do
    it "quantifies over variables only" do
      formula = p.call(x, y)
      expect { forall.call([x, c], formula) }.to raise_exception(TypeError)
      expect { forall.call(c, formula) }.to raise_exception(TypeError)
    end

    it "quantifies LTN objects only" do
      expect { forall.call([x, y], Torch.tensor([1, 2, 3])) }.to raise_exception(TypeError)
    end

    it "needs both or neither of cond_vars and cond_fn" do
      formula = p.call(x, y)
      expect { forall.call([x, y], formula, cond_vars: [x, y]) }.to raise_exception(ArgumentError, message: be =~ /cond_fn/)
      expect { forall.call([x, y], formula, cond_fn: sum_x_gt) }.to raise_exception(ArgumentError, message: be =~ /cond_vars/)
      expect { forall.call([x, y], formula) { |a| a } }.to raise_exception(ArgumentError, message: be =~ /cond_vars/)
    end

    it "takes the guard as cond_fn or as a block, not both" do
      expect { forall.call(x, p.call(x, y), cond_vars: x, cond_fn: sum_x_gt, &sum_x_gt) }
        .to raise_exception(ArgumentError, message: be =~ /both/)
    end

    it "checks the types of the guard" do
      formula = p.call(x, y)
      expect { forall.call(x, formula, cond_vars: c, cond_fn: sum_x_gt) }.to raise_exception(TypeError)
      expect { forall.call(x, formula, cond_vars: [x, c], cond_fn: sum_x_gt) }.to raise_exception(TypeError)
      expect { forall.call(x, formula, cond_vars: x, cond_fn: Torch.tensor([1, 2, 3])) }.to raise_exception(TypeError)
    end

    it "takes truth values in [0, 1]" do
      formula = LTN::LTNObject.new(Torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]), ["x", "y"])
      expect { forall.call([x, y], formula) }.to raise_exception(ArgumentError, message: be =~ /range/)
    end

    it "only quantifies variables of the formula" do
      z = LTN::Variable.new("z", Torch.randn(3, 2))
      expect { forall.call(z, p.call(x, y)) }.to raise_exception(ArgumentError, message: be =~ /"z" does not appear/)
    end
  end

  with "quantification" do
    it "aggregates the axis of one variable" do
      formula = p.call(x, y)

      output = forall.call(x, formula)
      expect(Torch.equal(output.value, pmean_error.call(formula.value, dim: 0))).to be == true
      expect(output.free_vars).to be == ["y"]
      expect(output.shape).to be == [4]

      output = forall.call(y, formula)
      expect(Torch.equal(output.value, pmean_error.call(formula.value, dim: 1))).to be == true
      expect(output.free_vars).to be == ["x"]
      expect(output.shape).to be == [2]
    end

    it "aggregates several variables into a closed formula" do
      formula = p.call(x, y)
      output = forall.call([x, y], formula)

      expect(Torch.equal(output.value, pmean_error.call(formula.value, dim: [0, 1]))).to be == true
      expect(output.free_vars).to be == []
      expect(output.shape).to be == []
    end

    it "nests" do
      formula = p.call(x, y)
      output = forall.call(x, exists.call(y, formula))
      expected = pmean_error.call(LTN::FuzzyOps::AggregPMean.new(p: 2).call(formula.value, dim: 1), dim: 0)

      expect(Torch.equal(output.value, expected)).to be == true
      expect(output.shape).to be == []
    end

    it "passes keyword arguments to the aggregator" do
      formula = p.call(x, y)

      expect(Torch.equal(forall.call(x, formula, p: 4).value, pmean_error.call(formula.value, dim: 0, p: 4))).to be == true
    end

    it "does not modify the formula" do
      formula = p.call(x, y)
      value = formula.value
      forall.call(y, formula, cond_vars: x, cond_fn: sum_x_gt)

      expect(formula.value).to be_equal(value)
      expect(formula.free_vars).to be == ["x", "y"]
    end
  end

  with "guarded quantification" do
    it "aggregates only where the guard holds, in float64" do
      formula = p.call(x, y)
      output = forall.call(x, formula, cond_vars: x, cond_fn: sum_x_gt)
      mask = Torch.tensor([[true] * 4, [false] * 4])

      expect(Torch.equal(output.value, pmean_error.call(formula.value, dim: 0, mask: mask).double)).to be == true
      expect(output.free_vars).to be == ["y"]
      expect(output.value.dtype).to be == :float64
    end

    it "takes the guard as a block" do
      output = forall.call(x, p.call(x, y), cond_vars: [x]) { |a| Torch.sum(a.value, dim: 1).gt(1.0) }

      expect(Torch.equal(output.value, forall.call(x, p.call(x, y), cond_vars: x, cond_fn: sum_x_gt).value)).to be == true
    end

    it "is vacuously true for forall, and false for exists, where the guard holds nowhere" do
      guard = ->(a, b) { Torch.logical_and(Torch.sum(a.value, dim: 1).gt(1.0), Torch.sum(b.value, dim: 1).gt(0.0)) }

      expect(forall.call(x, p.call(x, y), cond_vars: [x, y], cond_fn: guard).value.to_a.values_at(1, 2)).to be == [1.0, 1.0]
      expect(exists.call(x, p.call(x, y), cond_vars: [x, y], cond_fn: guard).value.to_a.values_at(1, 2)).to be == [0.0, 0.0]
    end

    it "resets diagonal quantification on the quantified variables" do
      v1 = LTN::Variable.new("v1", Torch.randn(3, 5))
      v2 = LTN::Variable.new("v2", Torch.randn(3, 6))
      output = forall.call(LTN.diag(v1, v2), p.call(v1, v2))

      expect(output.shape).to be == []
      expect(v1.free_vars).to be == ["v1"]
      expect(v2.free_vars).to be == ["v2"]
    end
  end

  with ".compute_mask" do
    let(:formula) { LTN::LTNObject.new(Torch.rand(2, 4), ["x", "y"]) }

    it "puts the guarded variables that are not quantified first" do
      f, m = subject.compute_mask(formula, [y], ->(b) { Torch.sum(b.value, dim: 1).lt(0.3) }, ["x"])

      expect(Torch.equal(f.value, formula.value.permute(1, 0))).to be == true
      expect(f.free_vars).to be == ["y", "x"]
      expect(m.free_vars).to be == ["y", "x"]
      expect(m.shape).to be == f.shape
      expect(m.value.to_a).to be == [[false, false], [true, true], [true, true], [false, false]]
    end

    it "keeps the order when all the guarded variables are quantified" do
      guard = ->(a, b) { Torch.sum(a.value, dim: 1).lt(Torch.sum(b.value, dim: 1)) }
      f, m = subject.compute_mask(formula, [x, y], guard, ["x", "y"])

      expect(Torch.equal(f.value, formula.value)).to be == true
      expect(f.free_vars).to be == ["x", "y"]
      expect(m.value.to_a).to be == [[false, false, false, false], [true, false, false, true]]
    end

    it "does not modify the formula (unlike LTNtorch)" do
      value = formula.value
      subject.compute_mask(formula, [y], ->(b) { Torch.sum(b.value, dim: 1).lt(0.3) }, ["x"])

      expect(formula.value).to be_equal(value)
      expect(formula.free_vars).to be == ["x", "y"]
    end
  end

  with ".transpose_vars" do
    let(:formula) { LTN::LTNObject.new(Torch.rand(3, 4), ["x", "y"]) }

    it "permutes the axes to the given order of variables, in a new object" do
      transposed = subject.transpose_vars(formula, ["y", "x"])

      expect(Torch.equal(transposed.value, formula.value.permute(1, 0))).to be == true
      expect(transposed.free_vars).to be == ["y", "x"]
      expect(formula.free_vars).to be == ["x", "y"]
    end

    it "leaves the axes as they are for the same order" do
      expect(Torch.equal(subject.transpose_vars(formula, ["x", "y"]).value, formula.value)).to be == true
    end
  end
end

# frozen_string_literal: true

require "ltn"
require_relative "../support/fixtures"

FUZZY_OPS_FIXTURE = Fixtures.load("fuzzy_ops")

module FuzzyOpsFixtureHelpers
  def build_operator(name, params)
    klass = LTN::FuzzyOps.const_get(name)
    return klass.new(build_operator(params["and_op"], {}), build_operator(params["implies_op"], {})) if name == "Equiv"

    klass.new(**params.transform_keys(&:to_sym))
  end

  def call_options(call)
    call.to_h do |key, value|
      value = Torch.tensor(value) if key == "mask"
      [key.to_sym, value]
    end
  end

  # Runs `operator` on fresh leaf tensors, then backpropagates the sum of its
  # output, as the fixture generator does.
  def run_case(operator, fixture_case)
    inputs = fixture_case["inputs"].map { |values| TorchTestHelpers.leaf_tensor(values) }
    output = operator.call(*inputs, **call_options(fixture_case["call"]))
    output.sum.backward if output.requires_grad
    [output, inputs.map(&:grad)]
  end

  def expect_case(output, grads, fixture_case)
    expected = fixture_case["output"]
    expect(Fixtures.matches?(output, expected)).to be_truthy
    if fixture_case["grads"].nil?
      expect(output.requires_grad).to be == false
    else
      fixture_case["grads"].zip(grads).each do |expected_grad, grad|
        if expected_grad.nil?
          expect(grad).to be_nil
        else
          expect(Fixtures.matches?(grad, expected_grad)).to be_truthy
        end
      end
    end
  end

  def self.label(fixture_case)
    details = fixture_case["params"].merge(fixture_case["call"].except("mask"))
    details["mask"] = true if fixture_case["call"].key?("mask")
    "#{fixture_case["op"]} #{details.map { |k, v| "#{k}=#{v.inspect}" }.join(" ")}".strip
  end
end

describe LTN::FuzzyOps do
  include FuzzyOpsFixtureHelpers

  it "was checked against reference values from the same PyTorch version as its LibTorch" do
    expect(FUZZY_OPS_FIXTURE["torch_version"]).to be == "2.15.0"
  end

  with "the projections" do
    let(:x) { Torch.tensor([0.0, 0.5, 1.0]) }

    it "pi_0 keeps truth values away from 0" do
      expect(TorchTestHelpers.all_close?(subject.pi_0(x), [1e-4, 0.5 * (1 - 1e-4) + 1e-4, 1.0], tolerance: 1e-7)).to be_truthy
    end

    it "pi_1 keeps truth values away from 1" do
      expect(TorchTestHelpers.all_close?(subject.pi_1(x), [0.0, 0.5 * (1 - 1e-4), 1 - 1e-4], tolerance: 1e-7)).to be_truthy
    end
  end

  with ".check_values" do
    it "accepts truth values in [0, 1]" do
      expect { subject.check_values(Torch.tensor([0.0, 1.0]), Torch.tensor(0.5)) }.not.to raise_exception
    end

    it "rejects values outside [0, 1]" do
      expect { subject.check_values(Torch.tensor([0.0, 1.0]), Torch.tensor([1.5])) }
        .to raise_exception(ArgumentError, message: be =~ /range \[0\., 1\.\]/)
      expect { subject.check_values(Torch.tensor([-0.1])) }.to raise_exception(ArgumentError)
    end

    it "rejects NaN" do
      expect { subject.check_values(Torch.tensor([Float::NAN])) }.to raise_exception(ArgumentError)
    end
  end

  with ".check_mask" do
    let(:xs) { Torch.tensor([[0.1, 0.2], [0.3, 0.4]]) }

    it "accepts a boolean mask shaped like the formula" do
      expect { subject.check_mask(Torch.tensor([[true, false], [false, true]]), xs) }.not.to raise_exception
    end

    it "rejects a mask with a different shape" do
      expect { subject.check_mask(Torch.tensor([true, false]), xs) }
        .to raise_exception(ArgumentError, message: be =~ /same shape/)
    end

    it "rejects a mask that is not boolean" do
      expect { subject.check_mask(Torch.tensor([[1, 0], [0, 1]]), xs) }
        .to raise_exception(ArgumentError, message: be =~ /boolean/)
    end
  end

  with "abstract operators" do
    [LTN::FuzzyOps::ConnectiveOperator, LTN::FuzzyOps::UnaryConnectiveOperator,
      LTN::FuzzyOps::BinaryConnectiveOperator, LTN::FuzzyOps::AggregationOperator].each do |klass|
      it "#{klass.name.split("::").last} requires subclasses to implement #call" do
        expect { klass.new.call(Torch.tensor([0.5])) }.to raise_exception(NotImplementedError, message: be =~ /#call/)
      end
    end
  end

  with "connective operators" do
    FUZZY_OPS_FIXTURE["connectives"].each do |fixture_case|
      it "#{FuzzyOpsFixtureHelpers.label(fixture_case)} matches LTNtorch, gradients included" do
        operator = build_operator(fixture_case["op"], fixture_case["params"])
        expect_case(*run_case(operator, fixture_case), fixture_case)
      end
    end

    it "can be called with the .() shorthand" do
      and_min = LTN::FuzzyOps::AndMin.new
      expect(and_min.(Torch.tensor([0.2]), Torch.tensor([0.7])).item).to be_within(1e-7).of(0.2) # standard:disable Style/LambdaCall
    end

    it "exposes and changes the stable flag" do
      and_prod = LTN::FuzzyOps::AndProd.new(stable: false)
      expect(and_prod.stable).to be == false
      and_prod.stable = true
      expect(and_prod.call(Torch.tensor([0.0]), Torch.tensor([0.0])).item).to be > 0.0
    end

    it "describes themselves" do
      expect(LTN::FuzzyOps::NotStandard.new.inspect).to be == "NotStandard()"
      expect(LTN::FuzzyOps::AndProd.new(stable: false).to_s).to be == "AndProd(stable: false)"
      equiv = LTN::FuzzyOps::Equiv.new(LTN::FuzzyOps::AndProd.new, LTN::FuzzyOps::ImpliesReichenbach.new)
      expect(equiv.inspect).to be == "Equiv(and_op: AndProd(stable: true), implies_op: ImpliesReichenbach(stable: true))"
      expect(equiv.and_op).to be_a(LTN::FuzzyOps::AndProd)
      expect(equiv.implies_op).to be_a(LTN::FuzzyOps::ImpliesReichenbach)
    end
  end

  with "aggregation operators" do
    FUZZY_OPS_FIXTURE["aggregators"].each do |fixture_case|
      it "#{FuzzyOpsFixtureHelpers.label(fixture_case)} matches LTNtorch, gradients included" do
        operator = build_operator(fixture_case["op"], fixture_case["params"])
        expect_case(*run_case(operator, fixture_case), fixture_case)
      end
    end

    it "accept a single integer dimension" do
      xs = Torch.tensor([[0.2, 0.4], [0.6, 0.8]])
      expect(TorchTestHelpers.all_close?(LTN::FuzzyOps::AggregMean.new.call(xs, dim: 1), [0.3, 0.7])).to be_truthy
    end

    it "reject a mask that does not match the formula" do
      xs = Torch.tensor([[0.2, 0.4], [0.6, 0.8]])
      [LTN::FuzzyOps::AggregMin, LTN::FuzzyOps::AggregMean, LTN::FuzzyOps::AggregPMean,
        LTN::FuzzyOps::AggregPMeanError].each do |klass|
        expect { klass.new.call(xs, dim: [1], mask: Torch.tensor([true, false])) }.to raise_exception(ArgumentError)
      end
    end

    it "describe themselves" do
      expect(LTN::FuzzyOps::AggregMin.new.inspect).to be == "AggregMin()"
      expect(LTN::FuzzyOps::AggregPMeanError.new(p: 3, stable: false).to_s)
        .to be == "AggregPMeanError(p: 3, stable: false)"
    end
  end
end

describe LTN::FuzzyOps::SatAgg do
  include FuzzyOpsFixtureHelpers

  FUZZY_OPS_FIXTURE["sat_agg"].each do |fixture_case|
    it "with #{fixture_case["agg_op"]} #{fixture_case["params"]} matches LTNtorch, gradients included" do
      sat_agg = subject.new(build_operator(fixture_case["agg_op"], fixture_case["params"]))
      inputs = fixture_case["inputs"].map { |value| TorchTestHelpers.leaf_tensor(value) }
      output = sat_agg.call(*inputs)
      output.backward

      expect(Fixtures.matches?(output, fixture_case["output"])).to be_truthy
      fixture_case["grads"].zip(inputs).each do |expected_grad, input|
        expect(Fixtures.matches?(input.grad, expected_grad)).to be_truthy
      end
    end
  end

  it "defaults to pMeanError with p = 2" do
    expect(subject.new.agg_op.inspect).to be == "AggregPMeanError(p: 2, stable: true)"
    expect(subject.new.to_s).to be == "SatAgg(agg_op: AggregPMeanError(p: 2, stable: true))"
  end

  it "accepts LTN objects as well as tensors" do
    formula = LTN::LTNObject.new(Torch.tensor(0.4), [])
    sat = subject.new(LTN::FuzzyOps::AggregMean.new).call(formula, Torch.tensor(0.8))

    expect(sat.item).to be_within(1e-6).of(0.6)
  end

  it "rejects an aggregator that is not an AggregationOperator" do
    expect { subject.new(LTN::FuzzyOps::AndProd.new) }.to raise_exception(TypeError, message: be =~ /AggregationOperator/)
  end

  it "rejects formulas that are neither LTN objects nor tensors" do
    expect { subject.new.call(Torch.tensor(0.5), 0.5) }.to raise_exception(TypeError, message: be =~ /LTNObjects and\/or tensors/)
  end

  it "rejects formulas that are not closed, reporting their shapes" do
    open_formula = LTN::LTNObject.new(Torch.tensor([0.5, 0.6]), ["x"])

    expect { subject.new.call(Torch.tensor(0.5), open_formula) }
      .to raise_exception(ArgumentError, message: be =~ /shapes: \[\[\], \[2\]\]/)
  end

  it "rejects truth values outside [0, 1]" do
    expect { subject.new.call(Torch.tensor(1.5)) }.to raise_exception(ArgumentError, message: be =~ /range/)
  end
end

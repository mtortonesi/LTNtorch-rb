# frozen_string_literal: true

require "ltn"

describe LTN::Connective do
  let(:not_standard) { subject.new(LTN::FuzzyOps::NotStandard.new) }
  let(:and_min) { subject.new(LTN::FuzzyOps::AndMin.new) }
  let(:op1) { LTN::LTNObject.new(Torch.rand(3, 4), ["x", "y"]) }
  let(:op2) { LTN::LTNObject.new(Torch.rand(5, 6), ["u", "z"]) }

  it "wants a connective operator" do
    expect { subject.new(LTN::FuzzyOps::AggregMean.new) }.to raise_exception(TypeError, message: be =~ /ConnectiveOperator/)
  end

  it "checks the number of operands" do
    expect { not_standard.call(op1, op2) }.to raise_exception(ArgumentError, message: be =~ /one operand/)
    expect { not_standard.call }.to raise_exception(ArgumentError)
    expect { and_min.call }.to raise_exception(ArgumentError, message: be =~ /two operands/)
    expect { and_min.call(op1, op2, op1) }.to raise_exception(ArgumentError)
  end

  it "takes LTN objects" do
    expect { and_min.call(op1, Torch.tensor([1, 2, 3, 4])) }.to raise_exception(TypeError)
  end

  it "takes truth values in [0, 1]" do
    wrong = LTN::LTNObject.new(Torch.tensor([[1.5, 0.2, 0.3]]), ["x"])

    expect { and_min.call(op1, wrong) }.to raise_exception(ArgumentError, message: be =~ /range/)
    expect { not_standard.call(wrong) }.to raise_exception(ArgumentError)
  end

  it "broadcasts operands with different free variables" do
    output = and_min.call(op1, op2)
    processed, = LTN.process_ltn_objects([op1, op2])

    expect(Torch.equal(Torch.flatten(output.value), Torch.minimum(processed[0].value, processed[1].value))).to be == true
    expect(output.free_vars).to be == ["x", "y", "u", "z"]
    expect(output.shape).to be == [3, 4, 5, 6]
  end

  it "applies the operator element-wise on operands with the same free variables" do
    a = LTN::LTNObject.new(Torch.rand(5), ["x"])
    b = LTN::LTNObject.new(Torch.rand(5), ["x"])
    output = and_min.call(a, b)

    expect(Torch.equal(output.value, Torch.minimum(a.value, b.value))).to be == true
    expect(output.free_vars).to be == ["x"]
    expect(output.shape).to be == [5]
  end

  it "passes keyword arguments to the operator (LTNtorch ignores them)" do
    and_prod = subject.new(LTN::FuzzyOps::AndProd.new)
    zero = LTN::LTNObject.new(Torch.tensor([0.0]), ["x"])

    expect(and_prod.call(zero, zero).value.item).to be > 0.0
    expect(and_prod.call(zero, zero, stable: false).value.item).to be == 0.0
  end

  it "keeps its operator and describes itself" do
    operator = LTN::FuzzyOps::AndProd.new(stable: false)
    connective = subject.new(operator)

    expect(connective.connective_op).to be_equal(operator)
    expect(connective.inspect).to be == "Connective(connective_op=AndProd(stable: false))"
    expect(not_standard.to_s).to be == "Connective(connective_op=NotStandard())"
  end
end

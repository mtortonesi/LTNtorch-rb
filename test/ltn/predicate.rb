# frozen_string_literal: true

require "ltn"

class SumPredicateModel < Torch::NN::Module
  def forward(x)
    Torch.sigmoid(Torch.sum(x, dim: 1))
  end
end

class BinarySumPredicateModel < Torch::NN::Module
  def forward(x, y)
    Torch.sigmoid(Torch.sum(Torch.cat([x, Torch.flatten(y, start_dim: 1)], dim: 1), dim: 1))
  end
end

describe LTN::Predicate do
  let(:x) { LTN::Variable.new("x", Torch.randn(4, 3)) }
  let(:y) { LTN::Variable.new("y", Torch.randn(5, 7)) }
  let(:c) { LTN::Constant.new(Torch.randn(5, 7)) }
  let(:sum_lambda) { ->(a) { Torch.sigmoid(Torch.sum(a, dim: 1)) } }

  with "construction" do
    it "needs a model, a function or a block" do
      expect { subject.new }.to raise_exception(ArgumentError, message: be =~ /None of/)
    end

    it "accepts only one of them" do
      expect { subject.new(SumPredicateModel.new, func: sum_lambda) }.to raise_exception(ArgumentError)
      expect { subject.new(func: sum_lambda) { |a| a } }.to raise_exception(ArgumentError)
      expect { subject.new(SumPredicateModel.new) { |a| a } }.to raise_exception(ArgumentError)
    end

    it "wants a Torch::NN::Module as model" do
      expect { subject.new(sum_lambda) }.to raise_exception(TypeError, message: be =~ /Torch::NN::Module/)
    end

    it "wants something callable, but not a module, as function" do
      expect { subject.new(func: SumPredicateModel.new) }.to raise_exception(TypeError, message: be =~ /function/)
      expect { subject.new(func: 42) }.to raise_exception(TypeError)
    end

    it "keeps the model" do
      model = SumPredicateModel.new
      predicate = subject.new(model)

      expect(predicate.model).to be_equal(model)
      expect(predicate.inspect).to be == "Predicate(model=SumPredicateModel())"
      expect(predicate.to_s).to be == predicate.inspect
    end

    it "wraps a function or block in a LambdaModel" do
      expect(subject.new(func: sum_lambda).model).to be_a(LTN::LambdaModel)
      expect(subject.new(func: method(:Integer)).model).to be_a(LTN::LambdaModel)
      expect(subject.new { |a| a }.model).to be_a(LTN::LambdaModel)
    end

    it "registers the model's parameters" do
      linear = Torch::NN::Linear.new(3, 1)
      predicate = subject.new(Torch::NN::Sequential.new(linear, Torch::NN::Sigmoid.new))

      expect(predicate.parameters.size).to be == 2
    end
  end

  with "a unary predicate" do
    def expect_unary(predicate, x)
      output = predicate.call(x)

      expect(output).to be_a(LTN::LTNObject)
      expect(Torch.equal(output.value.detach, Torch.sigmoid(Torch.sum(x.value, dim: 1)))).to be == true
      expect(output.shape).to be == [4]
      expect(output.free_vars).to be == ["x"]
    end

    it "works with a model" do
      expect_unary(subject.new(SumPredicateModel.new), x)
    end

    it "works with a function" do
      expect_unary(subject.new(func: sum_lambda), x)
    end

    it "works with a block" do
      expect_unary(subject.new { |a| Torch.sigmoid(Torch.sum(a, dim: 1)) }, x)
    end

    it "takes LTN objects, not tensors" do
      expect { subject.new(func: sum_lambda).call(Torch.tensor([1, 2, 3, 4])) }.to raise_exception(TypeError)
    end

    it "gives a scalar on a constant" do
      constant = LTN::Constant.new(Torch.randn(3, 2))
      output = subject.new { |a| Torch.sigmoid(Torch.sum(a)) }.call(constant)

      expect(output.value.item).to be_within(1e-6).of(Torch.sigmoid(Torch.sum(constant.value)).item)
      expect(output.shape).to be == []
      expect(output.free_vars).to be == []
    end
  end

  with "a binary predicate" do
    it "has one axis per variable" do
      output = subject.new(BinarySumPredicateModel.new).call(x, y)
      processed, = LTN.process_ltn_objects([x, y])
      expected = Torch.sigmoid(Torch.sum(Torch.cat([processed[0].value, processed[1].value], dim: 1), dim: 1)).view(4, 5)

      expect(Torch.equal(output.value.detach, expected)).to be == true
      expect(output.shape).to be == [4, 5]
      expect(output.free_vars).to be == ["x", "y"]
    end

    it "does not add an axis for a constant" do
      output = subject.new(BinarySumPredicateModel.new).call(x, c)

      expect(output.shape).to be == [4]
      expect(output.free_vars).to be == ["x"]
    end

    it "takes LTN objects, not tensors" do
      t = Torch.tensor([1, 2, 3, 4])
      expect { subject.new(BinarySumPredicateModel.new).call(t, t) }.to raise_exception(TypeError)
    end
  end

  it "passes keyword arguments to the model" do
    predicate = subject.new { |a, scale:| Torch.sigmoid(Torch.sum(a, dim: 1) * scale) }

    expect(predicate.call(x, scale: 0.0).value.to_a).to be == [0.5] * 4
  end

  it "rejects outputs outside [0, 1]" do
    predicate = subject.new { |a| Torch.sum(a, dim: 1) + 10.0 }

    expect { predicate.call(x) }.to raise_exception(ArgumentError, message: be =~ /range \[0\., 1\.\]/)
  end

  it "returns float32 truth values" do
    predicate = subject.new { |a| Torch.sigmoid(Torch.sum(a, dim: 1)).double }

    expect(predicate.call(x).value.dtype).to be == :float32
  end
end

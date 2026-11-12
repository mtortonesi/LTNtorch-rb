# frozen_string_literal: true

require "ltn"

class AppendOnesModel < Torch::NN::Module
  def forward(x)
    Torch.cat([x, Torch.ones(x.shape[0], 2)], dim: 1)
  end
end

describe LTN::Function do
  let(:x) { LTN::Variable.new("x", Torch.randn(4, 3)) }
  let(:y) { LTN::Variable.new("y", Torch.randn(5, 7)) }
  let(:c) { LTN::Constant.new(Torch.randn(5, 7)) }
  let(:append_ones) { ->(a) { Torch.cat([a, Torch.ones(a.shape[0], 2)], dim: 1) } }

  with "construction" do
    it "needs exactly one of a model, a function or a block" do
      expect { subject.new }.to raise_exception(ArgumentError)
      expect { subject.new(AppendOnesModel.new, func: append_ones) }.to raise_exception(ArgumentError)
    end

    it "checks the types of the model and the function" do
      expect { subject.new(append_ones) }.to raise_exception(TypeError)
      expect { subject.new(func: AppendOnesModel.new) }.to raise_exception(TypeError)
    end

    it "keeps the model, or wraps the function" do
      model = AppendOnesModel.new
      expect(subject.new(model).model).to be_equal(model)
      expect(subject.new(model).inspect).to be == "Function(model=AppendOnesModel())"
      expect(subject.new(func: append_ones).model).to be_a(LTN::LambdaModel)
    end
  end

  it "maps individuals to individuals, keeping the output's dimensions" do
    [subject.new(AppendOnesModel.new), subject.new(func: append_ones), subject.new(&append_ones)].each do |function|
      output = function.call(x)

      expect(Torch.equal(output.value.detach, Torch.cat([x.value, Torch.ones(4, 2)], dim: 1))).to be == true
      expect(output.shape).to be == [4, 5]
      expect(output.free_vars).to be == ["x"]
    end
  end

  it "can return one scalar per individual" do
    output = subject.new { |a| Torch.mean(a, dim: 1) }.call(x)

    expect(output.shape).to be == [4]
    expect(output.free_vars).to be == ["x"]
  end

  it "gives one value on a constant" do
    output = subject.new { |a| Torch.mean(a) }.call(LTN::Constant.new(Torch.randn(3, 2)))

    expect(output.shape).to be == []
    expect(output.free_vars).to be == []
  end

  it "has one leading axis per variable" do
    output = subject.new { |a, b| Torch.cat([a, b], dim: 1) }.call(x, y)
    processed, = LTN.process_ltn_objects([x, y])

    expect(Torch.equal(output.value, Torch.cat([processed[0].value, processed[1].value], dim: 1).view(4, 5, 10)))
      .to be == true
    expect(output.shape).to be == [4, 5, 10]
    expect(output.free_vars).to be == ["x", "y"]
  end

  it "does not add an axis for a constant" do
    output = subject.new { |a, b| Torch.cat([a, Torch.flatten(b, start_dim: 1)], dim: 1) }.call(x, c)

    expect(output.shape).to be == [4, 38]
    expect(output.free_vars).to be == ["x"]
  end

  it "takes LTN objects, not tensors" do
    expect { subject.new(func: append_ones).call(Torch.tensor([1, 2, 3, 4])) }.to raise_exception(TypeError)
  end

  it "returns float32 values" do
    expect(subject.new { |a| a.double }.call(x).value.dtype).to be == :float32
  end
end

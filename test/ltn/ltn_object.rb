# frozen_string_literal: true

require "ltn"

describe LTN::LTNObject do
  let(:value) { Torch.tensor([[0.1, 0.2], [0.3, 0.4]]) }

  it "stores a grounding and the labels of its free variables" do
    object = subject.new(value, ["x", "y"])

    expect(object.value).to be_equal(value)
    expect(object.free_vars).to be == ["x", "y"]
    expect(object.shape).to be == [2, 2]
  end

  it "accepts an object with no free variables" do
    object = subject.new(Torch.tensor(0.5), [])

    expect(object.free_vars).to be == []
    expect(object.shape).to be == []
  end

  it "rejects a value that is not a tensor" do
    expect { subject.new([0.1, 0.2], ["x"]) }.to raise_exception(TypeError, message: be =~ /must be a Torch::Tensor/)
  end

  it "rejects labels that are not an array of strings" do
    expect { subject.new(value, "x") }.to raise_exception(TypeError, message: be =~ /Array of String/)
    expect { subject.new(value, [:x]) }.to raise_exception(TypeError, message: be =~ /Array of String/)
  end

  it "shows its value and free variables" do
    object = subject.new(Torch.tensor([0.5]), ["x"])

    expect(object.inspect).to be =~ /\ALTNObject\(value=tensor\(\[0\.50*\]\), free_vars=\["x"\]\)\z/
    expect(object.to_s).to be == object.inspect
  end
end

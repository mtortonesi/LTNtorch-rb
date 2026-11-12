# frozen_string_literal: true

require "ltn"

describe LTN::Variable do
  let(:one_dim) { Torch.tensor([1, 2, 3, 4], dtype: :float64) }
  let(:more_dims) { Torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]) }

  it "has its label as only free variable" do
    variable = subject.new("x", more_dims)

    expect(variable).to be_a(LTN::LTNObject)
    expect(variable.free_vars).to be == ["x"]
    expect(variable.latent_var).to be == "x"
    expect(variable.shape).to be == [2, 4]
  end

  it "turns a 1-D tensor into one individual per row" do
    variable = subject.new("x", one_dim)

    expect(variable.shape).to be == [4, 1]
    expect(Torch.equal(variable.value, one_dim.float.unsqueeze(1))).to be == true
  end

  it "keeps a 1-D tensor when add_batch_dim is false" do
    expect(subject.new("x", one_dim, add_batch_dim: false).shape).to be == [4]
  end

  it "converts float64 individuals to float32" do
    expect(subject.new("x", one_dim).value.dtype).to be == :float32
  end

  it "keeps other dtypes" do
    expect(subject.new("x", more_dims).value.dtype).to be == :int64
  end

  it "lives on LTN.device" do
    expect(subject.new("x", one_dim).value.device.type).to be == LTN.device.type
  end

  it "rejects a label that is not a String" do
    expect { subject.new(1, more_dims) }.to raise_exception(TypeError)
    expect { subject.new(:x, more_dims) }.to raise_exception(TypeError)
  end

  it "reserves labels starting with diag_" do
    expect { subject.new("diag_x", more_dims) }.to raise_exception(ArgumentError, message: be =~ /reserved/)
  end

  it "rejects individuals that are not a tensor" do
    expect { subject.new("x", [1, 2, 3, 4]) }.to raise_exception(TypeError)
  end

  it "shows its value" do
    expect(subject.new("x", Torch.tensor([1, 2]), add_batch_dim: false).inspect)
      .to be == 'Variable(value=tensor([1, 2]), free_vars=["x"])'
  end
end

# frozen_string_literal: true

require "ltn"

describe LTN::Constant do
  let(:value) { Torch.tensor([1, 2, 3, 4]) }

  it "is an LTN object with no free variables" do
    constant = subject.new(value)

    expect(constant).to be_a(LTN::LTNObject)
    expect(constant.free_vars).to be == []
    expect(Torch.equal(constant.value, value)).to be == true
    expect(constant.shape).to be == [4]
  end

  it "is not trainable by default" do
    constant = subject.new(value)

    expect(constant.value.requires_grad).to be == false
    expect(constant.value.dtype).to be == :int64
  end

  it "lives on LTN.device" do
    expect(subject.new(value).value.device.type).to be == LTN.device.type
  end

  it "can be trainable, as a float32 leaf that requires grad" do
    constant = subject.new(value, trainable: true)

    expect(constant.value.dtype).to be == :float32
    expect(constant.value.requires_grad).to be == true
    expect(constant.value.leaf?).to be == true
  end

  it "receives gradients when trainable" do
    constant = subject.new(Torch.tensor([0.5, 2.0]), trainable: true)
    (constant.value * constant.value).sum.backward

    expect(constant.value.grad.to_a).to be == [1.0, 4.0]
  end

  it "rejects a value that is not a tensor" do
    expect { subject.new([1, 2, 3, 4]) }.to raise_exception(TypeError)
  end

  it "shows its value" do
    expect(subject.new(value).inspect).to be == "Constant(value=tensor([1, 2, 3, 4]), free_vars=[])"
    expect(subject.new(value, trainable: true).to_s)
      .to be == "Constant(value=tensor([1., 2., 3., 4.], requires_grad: true), free_vars=[])"
  end
end

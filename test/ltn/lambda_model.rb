# frozen_string_literal: true

require "ltn"

describe LTN::LambdaModel do
  it "is a module that calls its function" do
    model = subject.new(->(x) { Torch.sum(x) })
    t = Torch.tensor([1, 2, 3, 4, 5])

    expect(model).to be_a(Torch::NN::Module)
    expect(model.call(t).item).to be == 15
  end

  it "passes keyword arguments on" do
    model = subject.new(->(x, scale:) { x * scale })

    expect(model.call(Torch.tensor([1.0]), scale: 3.0).item).to be_within(1e-7).of(3.0)
  end

  it "has no parameters" do
    expect(subject.new(->(x) { x }).parameters).to be == []
  end
end

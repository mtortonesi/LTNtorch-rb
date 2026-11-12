# frozen_string_literal: true

require "ltn"

describe "LTN.diag and LTN.undiag" do
  let(:v1) { LTN::Variable.new("v1", Torch.tensor([[1.0, 2.0], [3.0, 4.0]])) }
  let(:v2) { LTN::Variable.new("v2", Torch.tensor([[5.0, 6.0], [7.0, 8.0]])) }
  let(:v3) { LTN::Variable.new("v3", Torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])) }
  let(:v4) { LTN::Variable.new("v4", Torch.tensor([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]])) }
  let(:c) { LTN::Constant.new(Torch.tensor([1.0, 2.0, 3.0, 4.0])) }
  let(:sum_all) { LTN::Function.new { |a, b, d| Torch.sum(Torch.cat([a, b, d], dim: 1), dim: 1) } }

  it "only accepts variables" do
    expect { LTN.diag(v1, c) }.to raise_exception(TypeError)
    expect { LTN.undiag(v1, c) }.to raise_exception(TypeError)
  end

  it "needs more than one variable" do
    expect { LTN.diag(v1) }.to raise_exception(ArgumentError, message: be =~ /just one/)
  end

  it "needs variables with the same number of individuals" do
    expect { LTN.diag(v1, v3) }.to raise_exception(ArgumentError, message: be =~ /\[2, 3\]/)
  end

  it "gives the variables a shared label, so they are paired element-wise" do
    expect(LTN.diag(v1, v2, v4)).to be == [v1, v2, v4]
    expect([v1, v2, v4].map(&:free_vars).uniq).to be == [["diag_v1_v2_v4"]]

    output = sum_all.call(v1, v2, v4)
    expect(Torch.equal(output.value, Torch.sum(Torch.cat([v1.value, v2.value, v4.value], dim: 1), dim: 1))).to be == true
    expect(output.free_vars).to be == ["diag_v1_v2_v4"]
    expect(output.shape).to be == [2]
  end

  it "is undone by undiag" do
    LTN.diag(v1, v2, v4)
    expect(LTN.undiag(v1, v2, v4)).to be == [v1, v2, v4]
    expect([v1, v2, v4].map(&:free_vars)).to be == [["v1"], ["v2"], ["v4"]]

    output = sum_all.call(v1, v2, v4)
    expect(output.free_vars).to be == ["v1", "v2", "v4"]
    expect(output.shape).to be == [2, 2, 2]
  end

  it "undiag leaves a variable that is not diagonal unchanged" do
    expect(LTN.undiag(v1).first.free_vars).to be == ["v1"]
  end
end

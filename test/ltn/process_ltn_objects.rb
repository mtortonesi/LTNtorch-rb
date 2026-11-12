# frozen_string_literal: true

require "ltn"

describe "LTN.process_ltn_objects" do
  let(:c1) { LTN::Constant.new(Torch.tensor([1, 2, 3, 4])) }
  let(:c2) { LTN::Constant.new(Torch.tensor([[2, 4, 3, 2], [4, 3, 2, 8]])) }
  let(:v1) { LTN::Variable.new("x", Torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])) }
  let(:v2) { LTN::Variable.new("y", Torch.tensor([[1, 2], [3, 4]])) }
  let(:v3) { LTN::Variable.new("z", Torch.tensor([[1, 2], [3, 4], [5, 6]])) }

  it "only accepts an array of LTN objects" do
    expect { LTN.process_ltn_objects([c1, Torch.tensor([3, 4, 5, 6])]) }.to raise_exception(TypeError)
    expect { LTN.process_ltn_objects(c1) }.to raise_exception(TypeError)
  end

  it "adds a batch dimension to constants, and no variables" do
    processed, vars, n_individuals = LTN.process_ltn_objects([c1, c2])

    expect(vars).to be == []
    expect(n_individuals).to be == []
    expect(processed[0].value.to_a).to be == [[1, 2, 3, 4]]
    expect(processed[1].value.squeeze.to_a).to be == [[2, 4, 3, 2], [4, 3, 2, 8]]
    expect(processed.map(&:free_vars)).to be == [[], []]
  end

  it "repeats a constant for each individual of a variable" do
    processed, vars, n_individuals = LTN.process_ltn_objects([c1, v1])

    expect(vars).to be == ["x"]
    expect(n_individuals).to be == [2]
    expect(processed[0].value.to_a).to be == [[1, 2, 3, 4], [1, 2, 3, 4]]
    expect(processed[1].value.to_a).to be == v1.value.to_a
    expect(processed.map(&:free_vars)).to be == [["x"], ["x"]]
  end

  it "pairs every individual of a variable with every individual of another" do
    processed, vars, n_individuals = LTN.process_ltn_objects([v1, v2])

    expect(vars).to be == ["x", "y"]
    expect(n_individuals).to be == [2, 2]
    expect(processed[0].value.to_a).to be == [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
      [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]
    expect(processed[1].value.to_a).to be == [[1, 2], [3, 4], [1, 2], [3, 4]]
    expect(processed.map(&:free_vars)).to be == [["x", "y"], ["x", "y"]]
  end

  it "handles variables with different numbers of individuals" do
    processed, vars, n_individuals = LTN.process_ltn_objects([v1, v3])

    expect(vars).to be == ["x", "z"]
    expect(n_individuals).to be == [2, 3]
    expect(processed[0].value.to_a).to be == [[1, 2, 3, 4, 5, 6]] * 3 + [[7, 8, 9, 10, 11, 12]] * 3
    expect(processed[1].value.to_a).to be == [[1, 2], [3, 4], [5, 6]] * 2
  end

  it "pairs diagonally quantified variables element-wise" do
    x = LTN::Variable.new("x", Torch.randn(3, 4))
    y = LTN::Variable.new("y", Torch.randn(3, 6))
    LTN.diag(x, y)
    processed, vars, n_individuals = LTN.process_ltn_objects([x, y])

    expect(Torch.equal(processed[0].value, x.value)).to be == true
    expect(Torch.equal(processed[1].value, y.value)).to be == true
    expect(vars).to be == ["diag_x_y"]
    expect(x.free_vars).to be == ["diag_x_y"]
    expect(n_individuals).to be == [3]
  end

  it "returns new objects" do
    processed, = LTN.process_ltn_objects([c1, v1])

    expect(processed[0]).not.to be_equal(c1)
    expect(processed[1]).not.to be_equal(v1)
  end

  it "does not modify its arguments, intermediate results included (unlike LTNtorch)" do
    formula = LTN::Predicate.new { |a| Torch.sigmoid(a.float.sum(1)) }.call(v1)
    value = formula.value
    LTN.process_ltn_objects([formula, v2])

    expect(formula.value).to be_equal(value)
    expect(formula.free_vars).to be == ["x"]
    expect(v2.free_vars).to be == ["y"]
  end

  it "keeps gradients flowing to trainable constants" do
    constant = LTN::Constant.new(Torch.tensor([1.0, 2.0]), trainable: true)
    processed, = LTN.process_ltn_objects([constant, LTN::Variable.new("x", Torch.randn(3, 2))])
    processed[0].value.sum.backward

    expect(constant.value.grad.to_a).to be == [3.0, 3.0]
  end
end

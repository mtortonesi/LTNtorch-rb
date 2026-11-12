# frozen_string_literal: true

# Trains a small knowledge base from the data and initial weights generated
# by test/fixtures/training.py, and checks that the loss follows LTNtorch's
# at every epoch: this covers the whole training loop (groundings, guarded
# quantification, SatAgg, autograd and Adam), not single evaluations.

require "ltn"
require_relative "../support/fixtures"

TRAINING_FIXTURE = Fixtures.load("training")

describe "LTN training parity with LTNtorch" do
  let(:model_class) do
    Class.new(Torch::NN::Module) do
      def initialize
        super
        @layer1 = Torch::NN::Linear.new(2, 16)
        @layer2 = Torch::NN::Linear.new(16, 16)
        @layer3 = Torch::NN::Linear.new(16, 1)
        @elu = Torch::NN::ELU.new
        @sigmoid = Torch::NN::Sigmoid.new
      end

      def forward(x)
        x = @elu.call(@layer1.call(x))
        x = @elu.call(@layer2.call(x))
        @sigmoid.call(@layer3.call(x))
      end
    end
  end

  it "follows LTNtorch's loss at every epoch" do
    model = model_class.new
    Torch.no_grad do
      TRAINING_FIXTURE["initial"].each do |name, values|
        layer, parameter = name.split(".")
        model.instance_variable_get(:"@#{layer}").public_send(parameter).copy!(Torch.tensor(values))
      end
    end

    a = LTN::Predicate.new(model)
    not_ = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
    forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
    sat_agg = LTN::FuzzyOps::SatAgg.new
    optimizer = Torch::Optim::Adam.new(a.parameters, lr: 0.01)
    data = Torch.tensor(TRAINING_FIXTURE["data"])
    labels = Torch.tensor(TRAINING_FIXTURE["labels"])

    losses = TRAINING_FIXTURE["losses"].map do
      optimizer.zero_grad
      x_a = LTN::Variable.new("x_A", data[labels])
      x_not_a = LTN::Variable.new("x_not_A", data[labels.logical_not])
      x = LTN::Variable.new("x", data)
      sat = sat_agg.call(
        forall.call(x_a, a.call(x_a)),
        forall.call(x_not_a, not_.call(a.call(x_not_a))),
        forall.call(x, not_.call(a.call(x)), cond_vars: [x]) { |u| Torch.sum(Torch.square(u.value - 0.5), dim: 1).gt(0.2) }
      )
      loss = 1.0 - sat
      loss.backward
      optimizer.step
      loss.item
    end

    worst = losses.zip(TRAINING_FIXTURE["losses"]).map { |actual, expected| (actual - expected).abs }.max
    inform("largest difference from LTNtorch's loss: #{worst}")
    expect(worst).to be < 1e-5
  end
end

#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/2-multi_class_single_label_classification.ipynb, copyright 2021-2024 by
# Tommaso Carraro.

# Multi-class single-label classification of the Iris dataset: learn a
# predicate P(x, l), "example x has label l", from the knowledge base
#
#   forall x_A: P(x_A, l_A)
#   forall x_B: P(x_B, l_B)
#   forall x_C: P(x_C, l_C)
#
# where l_A, l_B, l_C are one-hot constants for the three species. P is a
# softmax classifier, so each example has exactly one label.
#
#   bundle exec ruby examples/2_multi_class_single_label_classification.rb

require_relative "support/example"

module LTNExamples
  class MultiClassSingleLabelClassification
    # Returns the logits of the classes of x (the softmax is applied by
    # LogitsToPredicate, so that the logits can be used for accuracy).
    class MLP < Torch::NN::Module
      def initialize(layer_sizes = [4, 16, 16, 8, 3])
        super()
        @elu = Torch::NN::ELU.new
        @dropout = Torch::NN::Dropout.new(p: 0.2)
        @linear_layers = Torch::NN::ModuleList.new(
          layer_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
      end

      def forward(x, training: false)
        @linear_layers.to_a[0...-1].each do |layer|
          x = @elu.call(layer.call(x))
          x = @dropout.call(x) if training
        end
        @linear_layers.to_a.last.call(x)
      end
    end

    # The truth value of P(x, l): the softmax probability of the class l
    # (one-hot) of example x.
    class LogitsToPredicate < Torch::NN::Module
      def initialize(logits_model)
        super()
        @logits_model = logits_model
        @softmax = Torch::NN::Softmax.new(dim: 1)
      end

      def forward(x, l, training: false)
        probabilities = @softmax.call(@logits_model.call(x, training: training))
        Torch.sum(probabilities * l, dim: 1)
      end
    end

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      @train = load("iris_training.csv")
      @test = load("iris_test.csv")

      @l_a = LTN::Constant.new(Torch.tensor([1, 0, 0]))
      @l_b = LTN::Constant.new(Torch.tensor([0, 1, 0]))
      @l_c = LTN::Constant.new(Torch.tensor([0, 0, 1]))

      @mlp = MLP.new
      @p = LTN::Predicate.new(LogitsToPredicate.new(@mlp))
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
      @sat_agg = LTN::FuzzyOps::SatAgg.new
    end

    def run(epochs: 500, log_every: 20)
      optimizer = Torch::Optim::Adam.new(@p.parameters, lr: 0.001)

      epochs.times do |epoch|
        train_loss = 0.0
        batches = LTNExamples.batches(@train, batch_size: 64, shuffle: true)
        batches.each do |data, labels|
          optimizer.zero_grad
          loss = 1.0 - satisfaction(data, labels, training: true)
          loss.backward
          optimizer.step
          train_loss += loss.item
        end
        next unless (epoch % log_every).zero?

        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f",
          epoch, train_loss / batches.size, sat_level(*@train), sat_level(*@test), accuracy(*@train), accuracy(*@test))
      end

      {train_sat: sat_level(*@train), test_sat: sat_level(*@test),
       train_accuracy: accuracy(*@train), test_accuracy: accuracy(*@test)}
    end

    def satisfaction(data, labels, training: false)
      x_a = LTN::Variable.new("x_A", data[labels.eq(0)])
      x_b = LTN::Variable.new("x_B", data[labels.eq(1)])
      x_c = LTN::Variable.new("x_C", data[labels.eq(2)])
      @sat_agg.call(
        @forall.call(x_a, @p.call(x_a, @l_a, training: training)),
        @forall.call(x_b, @p.call(x_b, @l_b, training: training)),
        @forall.call(x_c, @p.call(x_c, @l_c, training: training))
      )
    end

    def sat_level(data, labels)
      Torch.no_grad { satisfaction(data, labels).item }
    end

    def accuracy(data, labels)
      predictions = Torch.no_grad { Torch.argmax(@mlp.call(data), dim: 1) }
      LTNExamples.accuracy(predictions, labels)
    end

    private

    def load(name)
      _, rows = LTNExamples.read_table(name)
      [Torch.tensor(rows.map { |row| row[0...4].map(&:to_f) }), Torch.tensor(rows.map { |row| Integer(row[4]) })]
    end
  end
end

LTNExamples.main(LTNExamples::MultiClassSingleLabelClassification.new, __FILE__)

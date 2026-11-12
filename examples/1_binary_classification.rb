#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/1-binary_classification.ipynb, copyright 2021-2024 by Tommaso Carraro.

# Binary classification: learn a predicate A(x) that holds for the points of
# [0, 1]^2 within distance 0.3 of (0.5, 0.5), from the knowledge base
#
#   forall x_A: A(x_A)
#   forall x_not_A: not A(x_not_A)
#
# where x_A and x_not_A are the positive and negative training examples.
#
#   bundle exec ruby examples/1_binary_classification.rb

require_relative "support/example"

module LTNExamples
  class BinaryClassification
    # The grounding of A: a multi-layer perceptron ending in a sigmoid.
    class ModelA < Torch::NN::Module
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

    attr_reader :a, :dataset, :labels

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      @dataset = Torch.rand(100, 2)
      @labels = Torch.sum(Torch.square(@dataset - 0.5), dim: 1).lt(0.09)

      @a = LTN::Predicate.new(ModelA.new)
      @not = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
      @sat_agg = LTN::FuzzyOps::SatAgg.new
    end

    # @returns [Hash] the final satisfaction levels and accuracies.
    def run(epochs: 1000, log_every: 20)
      train, test = [@dataset[0...50], @labels[0...50]], [@dataset[50..], @labels[50..]]
      optimizer = Torch::Optim::Adam.new(@a.parameters, lr: 0.001)

      epochs.times do |epoch|
        train_loss = 0.0
        batches = LTNExamples.batches(train, batch_size: 64, shuffle: true)
        batches.each do |data, labels|
          optimizer.zero_grad
          loss = 1.0 - satisfaction(data, labels)
          loss.backward
          optimizer.step
          train_loss += loss.item
        end
        next unless (epoch % log_every).zero?

        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | Test Acc %.3f",
          epoch, train_loss / batches.size, sat_level(*train), sat_level(*test), accuracy(*train), accuracy(*test))
      end

      {train_sat: sat_level(*train), test_sat: sat_level(*test),
       train_accuracy: accuracy(*train), test_accuracy: accuracy(*test)}
    end

    # The satisfaction level of the knowledge base on the given examples.
    def satisfaction(data, labels)
      x_a = LTN::Variable.new("x_A", data[labels])
      x_not_a = LTN::Variable.new("x_not_A", data[labels.logical_not])
      @sat_agg.call(
        @forall.call(x_a, @a.call(x_a)),
        @forall.call(x_not_a, @not.call(@a.call(x_not_a)))
      )
    end

    def sat_level(data, labels)
      Torch.no_grad { satisfaction(data, labels).item }
    end

    def accuracy(data, labels)
      predictions = Torch.no_grad { @a.model.call(data).squeeze(1).gt(0.5) }
      LTNExamples.accuracy(predictions, labels)
    end
  end
end

LTNExamples.main(LTNExamples::BinaryClassification.new, __FILE__) do |example|
  points, labels = example.dataset[50..], example.labels[50..]
  predicted = Torch.no_grad { example.a.model.call(points).squeeze(1).gt(0.5) }
  [["Ground truth (test data)", labels], ["A(x) > 0.5 (test data)", predicted]].each do |title, mask|
    puts
    puts LTNExamples.scatter({"A" => ["#", points[mask]], "not A" => [".", points[mask.logical_not]]}, title: title)
  end
end

#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/5-regression.ipynb, copyright 2021-2024 by Tommaso Carraro.

# Regression on the Real estate valuation dataset: learn a function f(x)
# predicting the price per unit area of a house from its features, from the
# knowledge base
#
#   forall diag(x, y): Eq(f(x), y)
#
# where Eq(u, v) = exp(-alpha * ||u - v||) is a smooth equality, and diag
# pairs each house x with its own price y (diagonal quantification).
#
#   bundle exec ruby examples/5_regression.rb

require_relative "support/example"

module LTNExamples
  class Regression
    # A multi-layer perceptron with a linear output.
    class MLP < Torch::NN::Module
      def initialize(layer_sizes = [6, 8, 8, 1])
        super()
        @elu = Torch::NN::ELU.new
        @linear_layers = Torch::NN::ModuleList.new(
          layer_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
      end

      def forward(x)
        @linear_layers.to_a[0...-1].each { |layer| x = @elu.call(layer.call(x)) }
        @linear_layers.to_a.last.call(x)
      end
    end

    FEATURES = ["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
      "X4 number of convenience stores", "X5 latitude", "X6 longitude"].freeze
    TARGET = "Y house price of unit area"
    ALPHA = 0.05

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      header, rows = LTNExamples.read_table("real-estate.csv")
      rows = rows.values_at(*Torch.randperm(rows.size).to_a) # shuffle, as LTNtorch does
      x = Torch.tensor(rows.map { |row| FEATURES.map { |column| Float(row[header.index(column)]) } })
      y = Torch.tensor(rows.map { |row| [Float(row[header.index(TARGET)])] })
      @train = [x[0...330], y[0...330]]
      @test = [x[330..], y[330..]]

      @f = LTN::Function.new(MLP.new)
      @eq = LTN::Predicate.new { |u, v| Torch.exp(Torch.sqrt(Torch.sum(Torch.square(u - v), dim: 1)) * -ALPHA) }
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
    end

    def run(epochs: 500, log_every: 50)
      optimizer = Torch::Optim::Adam.new(@f.parameters, lr: 0.0005)

      epochs.times do |epoch|
        train_loss = 0.0
        batches = LTNExamples.batches(@train, batch_size: 64, shuffle: true)
        batches.each do |x_data, y_data|
          optimizer.zero_grad
          loss = 1.0 - satisfaction(x_data, y_data)
          loss.backward
          optimizer.step
          train_loss += loss.item
        end
        next unless (epoch % log_every).zero?

        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train RMSE %.3f | Test RMSE %.3f",
          epoch, train_loss / batches.size, sat_level(*@train), sat_level(*@test), rmse(*@train), rmse(*@test))
      end

      {train_sat: sat_level(*@train), test_sat: sat_level(*@test), train_rmse: rmse(*@train), test_rmse: rmse(*@test)}
    end

    def satisfaction(x_data, y_data)
      x = LTN::Variable.new("x", x_data)
      y = LTN::Variable.new("y", y_data)
      @forall.call(LTN.diag(x, y), @eq.call(@f.call(x), y)).value
    end

    # LTNtorch averages over batches of 64; the whole set is a single batch for the test set.
    def sat_level(x_data, y_data)
      batches = LTNExamples.batches([x_data, y_data], batch_size: 64, shuffle: false)
      Torch.no_grad { batches.sum { |x, y| satisfaction(x, y).item } / batches.size }
    end

    def rmse(x_data, y_data)
      batches = LTNExamples.batches([x_data, y_data], batch_size: 64, shuffle: false)
      Torch.no_grad do
        batches.sum { |x, y| Torch.sqrt(Torch.mean(Torch.square(@f.model.call(x) - y))).item } / batches.size
      end
    end
  end
end

LTNExamples.main(LTNExamples::Regression.new, __FILE__)

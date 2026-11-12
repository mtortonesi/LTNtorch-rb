#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/3-multi_class_multi_label_classification.ipynb, copyright 2021-2024 by
# Tommaso Carraro.

# Multi-class multi-label classification of the Leptograpsus crabs dataset:
# each crab has a color (blue or orange) and a sex (male or female). Learn a
# predicate P(x, l), "crab x has label l", from the knowledge base
#
#   forall x_blue: P(x_blue, l_blue)        (and likewise for orange, male, female)
#   forall x: not (P(x, l_blue) and P(x, l_orange))
#   forall x: not (P(x, l_male) and P(x, l_female))
#
# P is a sigmoid classifier, so the labels are learned independently, and the
# last two axioms state that they are mutually exclusive. After training, the
# script also queries three formulas the knowledge base does not state:
#
#   phi1: forall x: P(x, l_blue) -> not P(x, l_orange)   (implied by the axioms)
#   phi2: forall x: P(x, l_blue) -> P(x, l_orange)       (contradicted by them)
#   phi3: forall x: P(x, l_blue) -> P(x, l_male)         (independent of them)
#
#   bundle exec ruby examples/3_multi_class_multi_label_classification.rb

require_relative "support/example"

module LTNExamples
  class MultiClassMultiLabelClassification
    # Returns the logits of the four labels of x.
    class MLP < Torch::NN::Module
      def initialize(layer_sizes = [5, 16, 16, 8, 4])
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

    # The truth value of P(x, l): the sigmoid probability of the label l
    # (one-hot) of example x.
    class LogitsToPredicate < Torch::NN::Module
      def initialize(logits_model)
        super()
        @logits_model = logits_model
        @sigmoid = Torch::NN::Sigmoid.new
      end

      def forward(x, l, training: false)
        Torch.sum(@sigmoid.call(@logits_model.call(x, training: training)) * l, dim: 1)
      end
    end

    # Label codes, as in LTNtorch: colors 0 (blue) and 1 (orange), sexes 2 (male) and 3 (female).
    CODES = {"B" => 0, "O" => 1, "M" => 2, "F" => 3}.freeze

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      header, rows = LTNExamples.read_table("crabs.dat", separator: /\s+/)
      rows = rows.values_at(*Torch.randperm(rows.size).to_a) # shuffle, as LTNtorch does
      features = Torch.tensor(rows.map { |row| %w[FL RW CL CW BD].map { |column| Float(row[header.index(column)]) } })
      sex = Torch.tensor(rows.map { |row| CODES.fetch(row[header.index("sex")]) })
      color = Torch.tensor(rows.map { |row| CODES.fetch(row[header.index("sp")]) })
      @train = [features[0...160], sex[0...160], color[0...160]]
      @test = [features[160..], sex[160..], color[160..]]

      @l_blue, @l_orange, @l_male, @l_female = Torch.eye(4).to_a.map { |onehot| LTN::Constant.new(Torch.tensor(onehot)) }

      @mlp = MLP.new
      @p = LTN::Predicate.new(LogitsToPredicate.new(@mlp))
      @not = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
      @and = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
      @implies = LTN::Connective.new(LTN::FuzzyOps::ImpliesReichenbach.new)
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
      @sat_agg = LTN::FuzzyOps::SatAgg.new
    end

    def run(epochs: 500, log_every: 20)
      optimizer = Torch::Optim::Adam.new(@p.parameters, lr: 0.001)

      epochs.times do |epoch|
        train_loss = 0.0
        batches = LTNExamples.batches(@train, batch_size: 64, shuffle: true)
        batches.each do |data, sex, color|
          optimizer.zero_grad
          loss = 1.0 - satisfaction(data, sex, color)
          loss.backward
          optimizer.step
          train_loss += loss.item
        end
        next unless (epoch % log_every).zero?

        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train Acc %.3f | " \
          "Test Acc %.3f | Test Sat Phi 1 %.3f | Test Sat Phi 2 %.3f | Test Sat Phi 3 %.3f",
          epoch, train_loss / batches.size, sat_level(*@train), sat_level(*@test), accuracy(*@train),
          accuracy(*@test), *(1..3).map { |i| phi(i, @test[0]) })
      end

      {train_sat: sat_level(*@train), test_sat: sat_level(*@test),
       train_accuracy: accuracy(*@train), test_accuracy: accuracy(*@test),
       phi1: phi(1, @test[0]), phi2: phi(2, @test[0]), phi3: phi(3, @test[0])}
    end

    def satisfaction(data, sex, color)
      x = LTN::Variable.new("x", data)
      x_blue = LTN::Variable.new("x_blue", data[color.eq(0)])
      x_orange = LTN::Variable.new("x_orange", data[color.eq(1)])
      x_male = LTN::Variable.new("x_male", data[sex.eq(2)])
      x_female = LTN::Variable.new("x_female", data[sex.eq(3)])
      @sat_agg.call(
        @forall.call(x_blue, @p.call(x_blue, @l_blue)),
        @forall.call(x_orange, @p.call(x_orange, @l_orange)),
        @forall.call(x_male, @p.call(x_male, @l_male)),
        @forall.call(x_female, @p.call(x_female, @l_female)),
        @forall.call(x, @not.call(@and.call(@p.call(x, @l_blue), @p.call(x, @l_orange)))),
        @forall.call(x, @not.call(@and.call(@p.call(x, @l_male), @p.call(x, @l_female))))
      )
    end

    def sat_level(data, sex, color)
      Torch.no_grad { satisfaction(data, sex, color).item }
    end

    # The truth value of one of the queries phi1, phi2, phi3 on the given examples.
    def phi(index, data)
      Torch.no_grad do
        x = LTN::Variable.new("x", data)
        blue = @p.call(x, @l_blue)
        consequent = {1 => -> { @not.call(@p.call(x, @l_orange)) }, 2 => -> { @p.call(x, @l_orange) },
                      3 => -> { @p.call(x, @l_male) }}.fetch(index).call
        @forall.call(x, @implies.call(blue, consequent), p: 5).value.item
      end
    end

    # One minus the Hamming loss of the four labels. Like LTNtorch, it
    # thresholds the logits (not the probabilities) at 0.5.
    def accuracy(data, sex, color)
      predictions = Torch.no_grad { @mlp.call(data).gt(0.5) }
      onehot = Torch.stack([color.eq(0), color.eq(1), sex.eq(2), sex.eq(3)], dim: -1)
      predictions.eq(onehot).float.mean.item
    end
  end
end

LTNExamples.main(LTNExamples::MultiClassMultiLabelClassification.new, __FILE__)

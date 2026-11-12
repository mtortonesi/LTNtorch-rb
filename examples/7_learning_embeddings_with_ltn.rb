#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/7-learning_embeddings_with_LTN.ipynb, copyright 2021-2024 by Tommaso Carraro.

# Learning embeddings: the "smokers, friends and cancer" problem. Fourteen
# people, in two groups (a-h and i-n), are grounded as trainable constants
# (embeddings) learned together with three predicates: Smokes(x),
# Cancer(x) and Friends(x, y). The knowledge base contains:
#
# - facts: some friendships, who smokes, and who has cancer (the latter only
#   for the first group); all other friendships within a group are false;
# - rules: friendship is irreflexive and symmetric, everybody has a friend,
#   friends of smokers smoke, smoking causes cancer, and not smoking
#   prevents it.
#
# After training, the script queries two formulas the knowledge base does
# not state:
#
#   phi1: forall p: Cancer(p) -> Smokes(p)
#   phi2: forall p, q: (Cancer(p) or Cancer(q)) -> Friends(p, q)
#
#   bundle exec ruby examples/7_learning_embeddings_with_ltn.rb

require_relative "support/example"

module LTNExamples
  class LearningEmbeddings
    # A multi-layer perceptron over the concatenation of its arguments.
    class MLP < Torch::NN::Module
      def initialize(layer_sizes)
        super()
        @elu = Torch::NN::ELU.new
        @sigmoid = Torch::NN::Sigmoid.new
        @linear_layers = Torch::NN::ModuleList.new(
          layer_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
      end

      def forward(*inputs)
        x = (inputs.size == 1) ? inputs.first : Torch.cat(inputs, dim: 1)
        @linear_layers.to_a[0...-1].each { |layer| x = @elu.call(layer.call(x)) }
        @sigmoid.call(@linear_layers.to_a.last.call(x))
      end
    end

    EMBEDDING_SIZE = 5
    GROUP1 = ("a".."h").to_a.freeze
    GROUP2 = ("i".."n").to_a.freeze
    FRIENDS = [%w[a b], %w[a e], %w[a f], %w[a g], %w[b c], %w[c d], %w[e f], %w[g h],
      %w[i j], %w[j m], %w[k l], %w[m n]].freeze
    SMOKES = %w[a e f g j n].freeze
    CANCER = %w[a e].freeze

    attr_reader :people

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      @people = (GROUP1 + GROUP2).to_h { |person| [person, LTN::Constant.new(Torch.rand(EMBEDDING_SIZE), trainable: true)] }

      @cancer = LTN::Predicate.new(MLP.new([5, 16, 16, 1]))
      @smokes = LTN::Predicate.new(MLP.new([5, 16, 16, 1]))
      @friends = LTN::Predicate.new(MLP.new([10, 16, 16, 1]))

      @and = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
      @or = LTN::Connective.new(LTN::FuzzyOps::OrProbSum.new)
      @not = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
      @implies = LTN::Connective.new(LTN::FuzzyOps::ImpliesReichenbach.new)
      @exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists)
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
      @sat_agg = LTN::FuzzyOps::SatAgg.new
    end

    def run(epochs: 1000, log_every: 20)
      parameters = @smokes.parameters + @friends.parameters + @cancer.parameters + @people.values.map(&:value)
      optimizer = Torch::Optim::Adam.new(parameters, lr: 0.001)
      sat = nil

      epochs.times do |epoch|
        optimizer.zero_grad
        sat = satisfaction(p_exists: (epoch <= 200) ? 1 : 6)
        loss = 1.0 - sat
        loss.backward
        optimizer.step
        next unless (epoch % log_every).zero?

        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f | Phi1 Sat %.3f | Phi2 Sat %.3f",
          epoch, loss.item, sat.item, phi1, phi2)
      end

      {train_sat: sat.item, phi1: phi1, phi2: phi2}
    end

    def satisfaction(p_exists:)
      g = @people
      # The embeddings change at each step, so the variables are rebuilt from them.
      x = LTN::Variable.new("x", Torch.stack(g.values.map(&:value)))
      y = LTN::Variable.new("y", Torch.stack(g.values.map(&:value)))
      not_friends = [GROUP1, GROUP2].flat_map do |group|
        group.combination(2).reject { |pair| FRIENDS.include?(pair) }
          .map { |a, b| @not.call(@friends.call(g[a], g[b])) }
      end

      @sat_agg.call(
        @sat_agg.call(*FRIENDS.map { |a, b| @friends.call(g[a], g[b]) }),
        @sat_agg.call(*not_friends),
        @sat_agg.call(*SMOKES.map { |a| @smokes.call(g[a]) }),
        @sat_agg.call(*(g.keys - SMOKES).map { |a| @not.call(@smokes.call(g[a])) }),
        @sat_agg.call(*CANCER.map { |a| @cancer.call(g[a]) }),
        @sat_agg.call(*(GROUP1 - CANCER).map { |a| @not.call(@cancer.call(g[a])) }),
        @forall.call(x, @not.call(@friends.call(x, x)), p: 5),
        @forall.call([x, y], @implies.call(@friends.call(x, y), @friends.call(y, x)), p: 5),
        @forall.call(x, @exists.call(y, @friends.call(x, y), p: p_exists)),
        @forall.call([x, y], @implies.call(@and.call(@friends.call(x, y), @smokes.call(x)), @smokes.call(y))),
        @forall.call(x, @implies.call(@smokes.call(x), @cancer.call(x))),
        @forall.call(x, @implies.call(@not.call(@smokes.call(x)), @not.call(@cancer.call(x))))
      )
    end

    def phi1
      Torch.no_grad do
        p = LTN::Variable.new("p", Torch.stack(@people.values.map(&:value)))
        @forall.call(p, @implies.call(@cancer.call(p), @smokes.call(p)), p: 5).value.item
      end
    end

    def phi2
      Torch.no_grad do
        p = LTN::Variable.new("p", Torch.stack(@people.values.map(&:value)))
        q = LTN::Variable.new("q", Torch.stack(@people.values.map(&:value)))
        @forall.call([p, q], @implies.call(@or.call(@cancer.call(p), @cancer.call(q)), @friends.call(p, q)), p: 5)
          .value.item
      end
    end

    # The learned truth values: Smokes and Cancer for each person, and Friends for each pair.
    def truth_values
      Torch.no_grad do
        p = LTN::Variable.new("p", Torch.stack(@people.values.map(&:value)))
        q = LTN::Variable.new("q", Torch.stack(@people.values.map(&:value)))
        {smokes: @smokes.call(p).value, cancer: @cancer.call(p).value, friends: @friends.call(p, q).value}
      end
    end
  end
end

LTNExamples.main(LTNExamples::LearningEmbeddings.new, __FILE__) do |example|
  names = example.people.keys
  values = example.truth_values
  puts "\n     Smokes Cancer   (facts: smokes #{LTNExamples::LearningEmbeddings::SMOKES.join}, " \
    "cancer #{LTNExamples::LearningEmbeddings::CANCER.join}, unknown for #{LTNExamples::LearningEmbeddings::GROUP2.join})"
  names.each_with_index do |name, i|
    puts format("  %s    %.2f   %.2f", name, values[:smokes][i].item, values[:cancer][i].item)
  end
  [LTNExamples::LearningEmbeddings::GROUP1, LTNExamples::LearningEmbeddings::GROUP2].each do |group|
    puts "\nFriends(row, column):"
    puts "     " + group.map { |name| name.center(5) }.join
    group.each do |row|
      puts "  #{row}  " + group.map { |column| format("%5.2f", values[:friends][names.index(row), names.index(column)].item) }.join
    end
  end
end

#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/6-clustering.ipynb, copyright 2021-2024 by Tommaso Carraro.

# Clustering, without labels: learn a predicate C(x, c), "point x belongs to
# cluster c", for four clusters of points, from the knowledge base
#
#   forall x: exists c: C(x, c)                          every point is in some cluster
#   forall c: exists x: C(x, c)                          no cluster is empty
#   forall c, x, y with dist(x, y) < 0.2: C(x, c) <-> C(y, c)     close points share clusters
#   forall c, x, y with dist(x, y) > 1.0: not (C(x, c) and C(y, c))  distant points do not
#
# The last two axioms use guarded quantification.
#
#   bundle exec ruby examples/6_clustering.rb

require_relative "support/example"

module LTNExamples
  class Clustering
    # The probability, from a softmax over the clusters, that point x belongs
    # to cluster c (one-hot).
    class MLP < Torch::NN::Module
      def initialize(layer_sizes = [2, 16, 16, 16, 4])
        super()
        @elu = Torch::NN::ELU.new
        @softmax = Torch::NN::Softmax.new(dim: 1)
        @linear_layers = Torch::NN::ModuleList.new(
          layer_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
      end

      def forward(x, c)
        @linear_layers.to_a[0...-1].each { |layer| x = @elu.call(layer.call(x)) }
        Torch.sum(@softmax.call(@linear_layers.to_a.last.call(x)) * c, dim: 1)
      end
    end

    CLUSTERS = 4
    POINTS_PER_CLUSTER = 50
    CLOSE_THRESHOLD = 0.2
    DISTANT_THRESHOLD = 1.0
    MARGIN = 0.2

    attr_reader :data, :c

    def initialize(seed: 2020, output: $stdout)
      @output = output
      LTNExamples.seed(seed)

      # One cluster per quadrant of [-1, 1]^2, its mean drawn uniformly within
      # the quadrant (minus a margin), its points normally around the mean
      # with standard deviation 0.1.
      quadrants = [[-1, -1], [0, -1], [-1, 0], [0, 0]]
      @data = Torch.cat(quadrants.map do |low|
        low = Torch.tensor(low, dtype: :float32) + MARGIN
        mean = low + Torch.rand(2) * (1 - 2 * MARGIN)
        mean + Torch.randn(POINTS_PER_CLUSTER, 2) * 0.1
      end)

      @cluster = LTN::Predicate.new(MLP.new)
      @c = LTN::Variable.new("c", Torch.eye(CLUSTERS))
      @x = LTN::Variable.new("x", @data)
      @y = LTN::Variable.new("y", @data)
      @th_close = LTN::Constant.new(Torch.tensor(CLOSE_THRESHOLD))
      @th_distant = LTN::Constant.new(Torch.tensor(DISTANT_THRESHOLD))

      @and = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
      @not = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
      @equiv = LTN::Connective.new(LTN::FuzzyOps::Equiv.new(LTN::FuzzyOps::AndProd.new,
        LTN::FuzzyOps::ImpliesReichenbach.new))
      @exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 1), :exists)
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 4), :forall)
      @sat_agg = LTN::FuzzyOps::SatAgg.new
    end

    def run(epochs: 1000, log_every: 100)
      optimizer = Torch::Optim::Adam.new(@cluster.parameters, lr: 0.001)
      sat = nil

      epochs.times do |epoch|
        optimizer.zero_grad
        sat = satisfaction(p_exists: (epoch <= 100) ? 1 : 6)
        loss = 1.0 - sat
        loss.backward
        optimizer.step
        @output.puts format(" epoch %d | loss %.4f | Train Sat %.3f", epoch, loss.item, sat.item) if (epoch % log_every).zero?
      end

      {train_sat: sat.item, clusters_used: assignments.to_a.uniq.size}
    end

    def satisfaction(p_exists:)
      dist = ->(u, v) { Torch.norm(u - v, 2, dim: 1).unsqueeze(1) }
      @sat_agg.call(
        @forall.call(@x, @exists.call(@c, @cluster.call(@x, @c), p: p_exists)),
        @forall.call(@c, @exists.call(@x, @cluster.call(@x, @c), p: p_exists)),
        @forall.call([@c, @x, @y], @equiv.call(@cluster.call(@x, @c), @cluster.call(@y, @c)),
          cond_vars: [@x, @y]) { |x, y| dist.call(x.value, y.value).lt(@th_close.value) },
        @forall.call([@c, @x, @y], @not.call(@and.call(@cluster.call(@x, @c), @cluster.call(@y, @c))),
          cond_vars: [@x, @y]) { |x, y| dist.call(x.value, y.value).gt(@th_distant.value) }
      )
    end

    # The most likely cluster of each point.
    def assignments
      Torch.no_grad do
        probabilities = (0...CLUSTERS).map do |i|
          @cluster.model.call(@data, Torch.eye(CLUSTERS)[i].unsqueeze(0).expand(@data.shape[0], CLUSTERS))
        end
        Torch.argmax(Torch.stack(probabilities, dim: 1), dim: 1)
      end
    end
  end
end

LTNExamples.main(LTNExamples::Clustering.new, __FILE__) do |example|
  markers = %w[o + x *]
  assigned = example.assignments
  series = markers.each_with_index.to_h { |marker, i| ["cluster #{i}", [marker, example.data[assigned.eq(i)]]] }
  puts
  puts LTNExamples.scatter(series, title: "Learned clusters", xlim: [-1.0, 1.0], ylim: [-1.0, 1.0])
end

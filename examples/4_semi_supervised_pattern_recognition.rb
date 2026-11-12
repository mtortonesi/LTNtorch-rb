#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch's examples/4-semi-supervised_pattern_recognition.ipynb, copyright 2021-2024 by
# Tommaso Carraro.

# Semi-supervised pattern recognition: MNIST digit addition, from the LTN
# paper. The training data are pairs of MNIST images labeled with the sum of
# their digits only, never with the digits themselves; a digit classifier
# Digit(x, d) is learned from the knowledge base
#
#   forall diag(x, y, n): exists d1, d2 with d1 + d2 = n: Digit(x, d1) and Digit(y, d2)
#
# In the multi-digit task, each operand is a two-digit number (two images),
# and the guard becomes 10 d1 + d2 + 10 d3 + d4 = n.
#
# Like LTNtorch's notebook, the models initialize their weights with
# Kaiming/Xavier and their biases with N(0, 1) (#init_weights). With that
# initialization, single-digit addition reaches about 48% test accuracy
# after 20 epochs, in LTNtorch as here (checked against LTNtorch with the
# same settings: 46.5% vs 46.4% after one epoch, 48.4% vs 48.3% after 20).
# With PyTorch's default initialization (skipping the init_weights calls),
# LTNtorch reaches about 98%, as reported in the LTN paper: pass
# `--default-init` to train that way.
#
# The first run downloads MNIST (about 11 MB) into examples/datasets/MNIST.
# Training follows LTNtorch's settings (20 epochs on 30,000 additions), which
# takes a while on a CPU; set EPOCHS to train for fewer epochs, and pass
# `multi` to run the multi-digit task instead of the single-digit one.
#
#   bundle exec ruby examples/4_semi_supervised_pattern_recognition.rb [single|multi] [--default-init]

require_relative "support/example"
require_relative "support/mnist"

module LTNExamples
  class SemiSupervisedPatternRecognition
    # Convolutional embeddings of MNIST images.
    class MNISTConv < Torch::NN::Module
      # @parameter init_weights [Boolean] whether to initialize the weights as
      #   LTNtorch's example does (see #init_weights), or keep PyTorch's defaults.
      def initialize(conv_channels_sizes: [1, 6, 16], kernel_sizes: [5, 5], linear_layers_sizes: [256, 100],
        init_weights: true)
        super()
        @conv_layers = Torch::NN::ModuleList.new(
          conv_channels_sizes.each_cons(2).zip(kernel_sizes).map { |(input, output), kernel| Torch::NN::Conv2d.new(input, output, kernel) }
        )
        @relu = Torch::NN::ReLU.new
        @tanh = Torch::NN::Tanh.new
        @maxpool = Torch::NN::MaxPool2d.new([2, 2])
        @linear_layers = Torch::NN::ModuleList.new(
          linear_layers_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
        @batch_norm_layers = Torch::NN::ModuleList.new(
          linear_layers_sizes.drop(1).map { |size| Torch::NN::BatchNorm1d.new(size) }
        )
        self.init_weights if init_weights
      end

      def forward(x)
        @conv_layers.each { |conv| x = @maxpool.call(@relu.call(conv.call(x))) }
        x = Torch.flatten(x, start_dim: 1)
        @linear_layers.to_a.zip(@batch_norm_layers.to_a).each do |linear, batch_norm|
          x = @tanh.call(batch_norm.call(linear.call(x)))
        end
        x
      end

      # Kaiming (convolutions) and Xavier (linear layers) weights, and N(0, 1) biases.
      def init_weights
        @conv_layers.each do |layer|
          Torch::NN::Init.kaiming_uniform!(layer.weight)
          Torch::NN::Init.normal!(layer.bias)
        end
        @linear_layers.each do |layer|
          Torch::NN::Init.xavier_uniform!(layer.weight)
          Torch::NN::Init.normal!(layer.bias)
        end
      end
    end

    # Returns the logits of the ten digits for an MNIST image.
    class SingleDigitClassifier < Torch::NN::Module
      # @parameter init_weights [Boolean] as for MNISTConv.
      def initialize(layers_sizes: [100, 84, 10], init_weights: true)
        super()
        @mnistconv = MNISTConv.new(init_weights: init_weights)
        @tanh = Torch::NN::Tanh.new
        @linear_layers = Torch::NN::ModuleList.new(
          layers_sizes.each_cons(2).map { |input, output| Torch::NN::Linear.new(input, output) }
        )
        @batch_norm_layers = Torch::NN::ModuleList.new(
          layers_sizes.drop(1).map { |size| Torch::NN::BatchNorm1d.new(size) }
        )
        self.init_weights if init_weights
      end

      def forward(x)
        x = @mnistconv.call(x)
        linear_layers = @linear_layers.to_a
        linear_layers[0...-1].each_with_index do |layer, i|
          x = @tanh.call(@batch_norm_layers.to_a[i].call(layer.call(x)))
        end
        linear_layers.last.call(x)
      end

      # Xavier weights and N(0, 1) biases for the linear layers.
      def init_weights
        @linear_layers.each do |layer|
          Torch::NN::Init.xavier_uniform!(layer.weight)
          Torch::NN::Init.normal!(layer.bias)
        end
      end
    end

    # The truth value of Digit(x, d): the softmax probability of digit d for image x.
    class LogitsToPredicate < Torch::NN::Module
      def initialize(logits_model)
        super()
        @logits_model = logits_model
        @softmax = Torch::NN::Softmax.new(dim: 1)
      end

      def forward(x, d)
        Torch.gather(@softmax.call(@logits_model.call(x)), 1, d)
      end
    end

    TASKS = {
      single: {operands: 2, train_examples: 30_000, test_examples: 5_000},
      multi: {operands: 4, train_examples: 15_000, test_examples: 2_500}
    }.freeze

    # @parameter task [Symbol] :single (one-digit operands) or :multi (two-digit operands).
    # @parameter data [Hash, nil] {train: [images, labels], test: [images, labels]}, with images
    #   as uint8 n x 28 x 28 tensors; by default, MNIST.
    # @parameter init_weights [Boolean] true to initialize the classifier as LTNtorch's example
    #   does (about 48% test accuracy), false for PyTorch's defaults (about 98%).
    def initialize(task: :single, seed: 2020, output: $stdout, data: nil, init_weights: true)
      @task = task
      @output = output
      LTNExamples.seed(seed)

      data ||= {train: MNIST.load(:train), test: MNIST.load(:test)}
      settings = TASKS.fetch(task)
      @train = additions(*data[:train], settings[:operands], settings[:train_examples])
      @test = additions(*data[:test], settings[:operands], settings[:test_examples])

      @digits = (1..settings[:operands]).map { |i| LTN::Variable.new("d_#{i}", Torch.arange(10)) }
      @cnn = SingleDigitClassifier.new(init_weights: init_weights)
      @digit = LTN::Predicate.new(LogitsToPredicate.new(@cnn))
      @and = LTN::Connective.new(LTN::FuzzyOps::AndProd.new)
      @exists = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMean.new(p: 2), :exists)
      @forall = LTN::Quantifier.new(LTN::FuzzyOps::AggregPMeanError.new(p: 2), :forall)
    end

    def run(epochs: 20, batch_size: 32)
      optimizer = Torch::Optim::Adam.new(@digit.parameters, lr: 0.001)
      metrics = nil

      epochs.times do |epoch|
        # The exponent of the existential quantifier grows during training.
        p = case epoch
        when 0...4 then 1
        when 4...8 then 2
        when 8...12 then 4
        else 6
        end

        train = LTNExamples.batches(@train, batch_size: batch_size, shuffle: true).map do |images, labels|
          optimizer.zero_grad
          sat = satisfaction(images, labels, p)
          loss = 1.0 - sat
          loss.backward
          optimizer.step
          [loss.item, sat.item, accuracy(images, labels)]
        end
        test = LTNExamples.batches(@test, batch_size: batch_size, shuffle: false).map do |images, labels|
          Torch.no_grad do
            sat = satisfaction(images, labels, p)
            [1.0 - sat.item, sat.item, accuracy(images, labels)]
          end
        end

        train_loss, train_sat, train_acc = train.transpose.map { |values| values.sum / values.size }
        test_loss, test_sat, test_acc = test.transpose.map { |values| values.sum / values.size }
        metrics = {train_sat: train_sat, test_sat: test_sat, train_accuracy: train_acc, test_accuracy: test_acc}
        @output.puts format(" epoch %d | Train loss %.4f | Train Sat %.4f | Train Acc %.4f | Test loss %.4f | " \
          "Test Sat %.4f | Test Acc %.4f", epoch, train_loss, train_sat, train_acc, test_loss, test_sat, test_acc)
      end

      metrics
    end

    def satisfaction(images, labels, p)
      operands = images.shape[1].times.map { |i| LTN::Variable.new("x#{i + 1}", images[0.., i]) }
      z = LTN::Variable.new("z", labels)
      # diag must come first: it changes the labels of the variables, so the
      # formulas built from them below pair the operands with their sum.
      quantified = LTN.diag(*operands, z)
      digits = operands.zip(@digits).map { |x, d| @digit.call(x, d) }
      @forall.call(
        quantified,
        @exists.call(@digits, digits.reduce { |a, b| @and.call(a, b) },
          cond_vars: @digits + [z], cond_fn: ->(*ds, n) { number(ds.map(&:value)).eq(n.value) }, p: p)
      ).value
    end

    # Accuracy of the predicted sums.
    def accuracy(images, labels)
      Torch.no_grad do
        predictions = images.shape[1].times.map { |i| Torch.argmax(@cnn.call(images[0.., i]), dim: 1) }
        LTNExamples.accuracy(number(predictions), labels)
      end
    end

    private

    # The sum the digits stand for: d1 + d2, or (10 d1 + d2) + (10 d3 + d4).
    def number(digits)
      (@task == :single) ? digits[0] + digits[1] : digits[0] * 10 + digits[1] + digits[2] * 10 + digits[3]
    end

    # Groups consecutive slices of the images into additions, as LTNtorch does:
    # operand i of example j is image i * examples + j. With fewer images than
    # needed, makes as many additions as they allow.
    def additions(images, labels, operands, examples)
      examples = [examples, images.shape[0] / operands].min
      images = images.float / 255.0
      slices = (0...operands).map { |i| (i * examples)...((i + 1) * examples) }
      inputs = Torch.stack(slices.map { |range| images[range].unsqueeze(1) }, dim: 1)
      digits = slices.map { |range| labels[range] }
      [inputs, number(digits)]
    end
  end
end

if File.expand_path($PROGRAM_NAME) == File.expand_path(__FILE__)
  unknown = ARGV - %w[single multi --default-init]
  abort "Unknown arguments: #{unknown.join(" ")}\nUsage: #{$PROGRAM_NAME} [single|multi] [--default-init]" if unknown.any?

  example = LTNExamples::SemiSupervisedPatternRecognition.new(
    task: ARGV.include?("multi") ? :multi : :single,
    init_weights: !ARGV.include?("--default-init")
  )
  LTNExamples.main(example, __FILE__)
end

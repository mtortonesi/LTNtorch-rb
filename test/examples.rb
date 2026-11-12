# frozen_string_literal: true

# Smoke tests of the examples and tutorials: each runs briefly, so that they
# keep working as the library changes. benchmark/examples.rb checks that the
# examples still learn.

require "rbconfig"
require "stringio"
require_relative "../examples/1_binary_classification"
require_relative "../examples/2_multi_class_single_label_classification"
require_relative "../examples/3_multi_class_multi_label_classification"
require_relative "../examples/4_semi_supervised_pattern_recognition"
require_relative "../examples/5_regression"
require_relative "../examples/6_clustering"
require_relative "../examples/7_learning_embeddings_with_ltn"

describe "the examples" do
  def expect_metrics(metrics, keys)
    expect(metrics.keys).to be == keys
    metrics.each_value { |value| expect(value).to be_a(Numeric) }
    expect(metrics.values.all? { |value| value.to_f.finite? }).to be == true
  end

  it "run binary classification" do
    metrics = LTNExamples::BinaryClassification.new(output: StringIO.new).run(epochs: 2)
    expect_metrics(metrics, [:train_sat, :test_sat, :train_accuracy, :test_accuracy])
  end

  it "run multi-class single-label classification" do
    metrics = LTNExamples::MultiClassSingleLabelClassification.new(output: StringIO.new).run(epochs: 2)
    expect_metrics(metrics, [:train_sat, :test_sat, :train_accuracy, :test_accuracy])
  end

  it "run multi-class multi-label classification" do
    metrics = LTNExamples::MultiClassMultiLabelClassification.new(output: StringIO.new).run(epochs: 2)
    expect_metrics(metrics, [:train_sat, :test_sat, :train_accuracy, :test_accuracy, :phi1, :phi2, :phi3])
  end

  [[:single, true], [:multi, true], [:single, false]].each do |task, init_weights|
    it "run #{task}-digit MNIST addition, on random images, with init_weights: #{init_weights}" do
      images = ->(n) { Torch.randint(0, 256, [n, 28, 28], dtype: :uint8) }
      digits = ->(n) { Torch.randint(0, 10, [n], dtype: :int64) }
      data = {train: [images.call(128), digits.call(128)], test: [images.call(64), digits.call(64)]}
      example = LTNExamples::SemiSupervisedPatternRecognition.new(task: task, output: StringIO.new, data: data,
        init_weights: init_weights)

      expect_metrics(example.run(epochs: 1, batch_size: 16), [:train_sat, :test_sat, :train_accuracy, :test_accuracy])
    end
  end

  it "initialize the MNIST classifier's biases with N(0, 1) only with init_weights" do
    classifier = LTNExamples::SemiSupervisedPatternRecognition::SingleDigitClassifier
    bias_std = ->(model) { model.named_parameters.select { |name, _| name.end_with?("linear_layers.0.bias") }.values.map { |b| b.std.item } }

    expect(bias_std.call(classifier.new(init_weights: true)).min).to be > 0.5
    expect(bias_std.call(classifier.new(init_weights: false)).max).to be < 0.2
  end

  it "run regression" do
    metrics = LTNExamples::Regression.new(output: StringIO.new).run(epochs: 2)
    expect_metrics(metrics, [:train_sat, :test_sat, :train_rmse, :test_rmse])
  end

  it "run clustering" do
    metrics = LTNExamples::Clustering.new(output: StringIO.new).run(epochs: 2)
    expect_metrics(metrics, [:train_sat, :clusters_used])
  end

  it "run learning embeddings" do
    example = LTNExamples::LearningEmbeddings.new(output: StringIO.new)
    expect_metrics(example.run(epochs: 2), [:train_sat, :phi1, :phi2])
    expect(example.truth_values[:friends].shape).to be == [14, 14]
  end

  it "draw text scatter plots" do
    plot = LTNExamples.scatter({"A" => ["#", Torch.tensor([[0.0, 0.0]])], "B" => [".", Torch.tensor([[1.0, 1.0]])]},
      title: "Test", width: 4, height: 2)

    expect(plot).to be == "Test\n+----+\n|   .|\n|#   |\n+----+\nx: 0.0..1.0, y: 0.0..1.0; # A, . B"
  end
end

describe "the tutorials" do
  Dir[File.expand_path("../tutorials/*.rb", __dir__)].sort.each do |tutorial|
    it "run #{File.basename(tutorial)}" do
      environment = {"EPOCHS" => "1", "BATCH_EPOCHS" => "1"}
      output = IO.popen(environment, [RbConfig.ruby, "-I", File.expand_path("../lib", __dir__), tutorial],
        err: [:child, :out], &:read)

      inform(output) unless $?.success?
      expect($?.success?).to be == true
    end
  end
end

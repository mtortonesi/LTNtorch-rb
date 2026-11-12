# frozen_string_literal: true

# Learning benchmark: trains the examples with LTNtorch's settings and a
# fixed seed, and checks that they still reach the results of LTNtorch's
# notebooks, give or take. It takes about 20 seconds, so it is kept out of
# the default `sus` run; CI runs it as a separate job:
#
#   bundle exec sus benchmark/examples.rb
#
# The thresholds sit a little below the results of seed 2020 (in comments),
# to absorb numerical differences across platforms. Example 1 learns less
# with this seed than in LTNtorch's notebook (an unseeded run); trained from
# the same data and weights, the Ruby and Python versions follow the same
# loss (see test/ltn/training_parity.rb). Example 4 (MNIST) is left out: it
# needs a download and minutes of training.

require "stringio"
require_relative "../examples/1_binary_classification"
require_relative "../examples/2_multi_class_single_label_classification"
require_relative "../examples/3_multi_class_multi_label_classification"
require_relative "../examples/5_regression"
require_relative "../examples/6_clustering"
require_relative "../examples/7_learning_embeddings_with_ltn"

describe "the examples' learning" do
  def run(example_class)
    metrics = example_class.new(seed: 2020, output: StringIO.new).run
    inform(metrics.transform_values { |value| value.round(3) }.inspect)
    metrics
  end

  it "classifies points in a circle" do
    metrics = run(LTNExamples::BinaryClassification) # train_sat 0.706, train_accuracy 0.88
    expect(metrics[:train_sat]).to be > 0.65
    expect(metrics[:train_accuracy]).to be > 0.8
  end

  it "classifies the Iris dataset" do
    metrics = run(LTNExamples::MultiClassSingleLabelClassification) # train_sat 0.870, test_accuracy 0.967
    expect(metrics[:train_sat]).to be > 0.8
    expect(metrics[:test_accuracy]).to be > 0.9
  end

  it "classifies the crabs dataset, and answers queries" do
    metrics = run(LTNExamples::MultiClassMultiLabelClassification) # train_sat 0.934, test_accuracy 0.962
    expect(metrics[:train_sat]).to be > 0.85
    expect(metrics[:test_accuracy]).to be > 0.9
    expect(metrics[:phi1]).to be > 0.8 # implied by the knowledge base (0.871)
    expect(metrics[:phi2]).to be < 0.3 # contradicted by it (0.135)
  end

  it "predicts real estate prices" do
    metrics = run(LTNExamples::Regression) # train_sat 0.714, test_rmse 6.42
    expect(metrics[:train_sat]).to be > 0.65
    expect(metrics[:test_rmse]).to be < 9.0
  end

  it "clusters points" do
    metrics = run(LTNExamples::Clustering) # train_sat 0.854, 4 clusters
    expect(metrics[:train_sat]).to be > 0.8
    expect(metrics[:clusters_used]).to be == 4
  end

  it "learns embeddings of smokers and friends" do
    metrics = run(LTNExamples::LearningEmbeddings) # train_sat 0.841, phi1 0.947
    expect(metrics[:train_sat]).to be > 0.8
    expect(metrics[:phi1]).to be > 0.9
  end
end

# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.

require "ltn"

# Shared helpers of the examples, which are ports of LTNtorch's
# (https://github.com/tommasocarraro/LTNtorch/tree/main/examples).
#
# Each example is a class whose #run trains the model and returns its final
# metrics, so that benchmark/examples.rb and test/examples.rb can run it with
# fewer epochs; run as a script, it trains with LTNtorch's settings and
# prints its progress.
module LTNExamples
  DATASETS = File.expand_path("../datasets", __dir__)

  module_function

  # Mini-batches of rows of `tensors`, which share their first dimension, as
  # LTNtorch's examples' DataLoader yields them.
  #
  # @parameter tensors [Array(Torch::Tensor)] the data, e.g. features and labels.
  # @parameter batch_size [Integer] the number of rows per batch (the last batch may be smaller).
  # @parameter shuffle [Boolean] whether to shuffle the rows, with torch's RNG.
  # @returns [Array(Array(Torch::Tensor))] one array of tensors per batch.
  def batches(tensors, batch_size:, shuffle:)
    n = tensors.first.shape[0]
    order = shuffle ? Torch.randperm(n) : Torch.arange(n)
    (0...n).step(batch_size).map do |start|
      indices = order[start...[start + batch_size, n].min]
      tensors.map { |tensor| tensor[indices] }
    end
  end

  # Reads a table of numbers from a text file with a header line.
  #
  # @parameter name [String] the file name, in examples/datasets.
  # @parameter separator [String, Regexp] the column separator.
  # @returns [Array(Array(String), Array(Array(String)))] the header and the rows, as strings.
  def read_table(name, separator: ",")
    header, *rows = File.readlines(File.join(DATASETS, name), chomp: true).reject(&:empty?)
    split = ->(line) { line.strip.split(separator) }
    [split.call(header), rows.map(&split)]
  end

  # Seeds Ruby's and torch's random number generators.
  def seed(seed)
    srand(seed)
    Torch.manual_seed(seed)
  end

  # Fraction of equal elements of two tensors.
  def accuracy(predictions, labels)
    predictions.eq(labels).float.mean.item
  end

  # A scatter plot in plain text, each series drawn with its own character,
  # so it reads the same with or without colors.
  #
  # @parameter series [Hash(String, Array)] for each series name, its marker
  #   character and an n x 2 tensor of points, e.g. `{"A" => ["#", points]}`.
  # @parameter title [String] the title.
  # @parameter xlim [Array(Float)] the range of x.
  # @parameter ylim [Array(Float)] the range of y.
  # @returns [String] the plot.
  def scatter(series, title:, xlim: [0.0, 1.0], ylim: [0.0, 1.0], width: 48, height: 18)
    grid = Array.new(height) { Array.new(width, " ") }
    series.each_value do |marker, points|
      points.to_a.each do |x, y|
        column = ((x - xlim[0]) / (xlim[1] - xlim[0]) * (width - 1)).round
        row = ((ylim[1] - y) / (ylim[1] - ylim[0]) * (height - 1)).round
        grid[row][column] = marker if row.between?(0, height - 1) && column.between?(0, width - 1)
      end
    end
    legend = series.map { |name, (marker, _)| "#{marker} #{name}" }.join(", ")
    border = "+#{"-" * width}+"
    [title, border, *grid.map { |row| "|#{row.join}|" }, border,
      format("x: %.1f..%.1f, y: %.1f..%.1f; %s", *xlim, *ylim, legend)].join("\n")
  end

  # Runs `example` with the defaults of LTNtorch's version when the file is
  # executed as a script, then yields it (e.g. to plot its results).
  def main(example, file, **options)
    return unless File.expand_path($PROGRAM_NAME) == File.expand_path(file)

    epochs = ENV["EPOCHS"]&.then { |value| Integer(value) }
    metrics = epochs ? example.run(epochs: epochs, **options) : example.run(**options)
    puts "Final metrics: #{metrics.transform_values { |v| v.is_a?(Float) ? v.round(3) : v }}"
    yield example if block_given?
  end
end

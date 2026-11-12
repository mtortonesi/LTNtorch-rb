# frozen_string_literal: true

require "json"
require "torch"

# Reference values generated from LTNtorch by the scripts in test/fixtures
# (see the header of each script for how to regenerate them).
module Fixtures
  module_function

  DIRECTORY = File.expand_path("../fixtures", __dir__)

  SPECIAL_FLOATS = {"NaN" => Float::NAN, "Infinity" => Float::INFINITY, "-Infinity" => -Float::INFINITY}.freeze

  def load(name)
    decode(JSON.parse(File.read(File.join(DIRECTORY, "#{name}.json"))))
  end

  # The fixtures store non-finite floats as strings, to stay strict JSON.
  def decode(value)
    case value
    when Array then value.map { |v| decode(v) }
    when Hash then value.transform_values { |v| decode(v) }
    when String then SPECIAL_FLOATS.fetch(value, value)
    else value
    end
  end

  # Whether `tensor` matches a fixture's `{"shape", "dtype", "values"}` record
  # in shape, dtype and values (NaN matching NaN), within `tolerance`.
  def matches?(tensor, expected, tolerance: 1e-6)
    return false unless tensor.shape == expected["shape"]
    return false unless tensor.dtype.to_s == expected["dtype"]

    actual = tensor.shape.empty? ? [tensor.item] : tensor.to_a.flatten
    reference = Array(expected["values"]).flatten
    actual.size == reference.size && actual.zip(reference).all? { |a, e| close?(a, e, tolerance) }
  end

  def close?(actual, expected, tolerance)
    return actual == expected if actual == true || actual == false
    return actual.nan? if expected.nan?
    return actual == expected if expected.infinite?

    (actual - expected).abs <= tolerance
  end

  # Human-readable description of a mismatch, for test failure messages.
  def describe(tensor)
    tensor.nil? ? "nil" : "#{tensor.dtype} #{tensor.shape.inspect} #{tensor.to_a.inspect}"
  end
end

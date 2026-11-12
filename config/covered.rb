# frozen_string_literal: true

# Coverage measures the library: the examples, tutorials and benchmarks run
# in the test suite only briefly (their command-line entry points and the
# MNIST download do not run at all), and are checked by their own tests.
def ignore_paths
  super + ["examples/", "tutorials/", "benchmark/"]
end

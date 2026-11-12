# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

require "torch"

require_relative "ltn/version"

# Logic Tensor Networks (LTN) on top of torch.rb: a Ruby port of LTNtorch
# (https://github.com/tommasocarraro/LTNtorch).
module LTN
  class Error < StandardError; end

  class << self
    # The device on which LTN places the groundings of its variables.
    #
    # LTNtorch picks CUDA at import time when available. Here the default is
    # chosen lazily, on first access, and can be overridden at any time, e.g.
    # `LTN.device = Torch.device("cpu")` to force CPU in tests.
    #
    # @returns [Torch::Device]
    def device
      @device ||= Torch.device(Torch::CUDA.available? ? "cuda" : "cpu")
    end

    # @parameter device [Torch::Device, String] e.g. `"cpu"` or `"cuda:0"`.
    def device=(device)
      @device = device.is_a?(Torch::Device) ? device : Torch.device(device)
    end
  end
end

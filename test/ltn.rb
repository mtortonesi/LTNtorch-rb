# frozen_string_literal: true

require "ltn"

describe LTN do
  it "has a semantic version number" do
    expect(LTN::VERSION).to be =~ /\A\d+\.\d+\.\d+/
  end

  with ".device" do
    def restore_device
      previous = LTN.instance_variable_get(:@device)
      yield
    ensure
      LTN.instance_variable_set(:@device, previous)
    end

    it "defaults to CUDA when available and to the CPU otherwise" do
      restore_device do
        LTN.instance_variable_set(:@device, nil)
        expected = Torch::CUDA.available? ? "cuda" : "cpu"

        expect(LTN.device).to be_a(Torch::Device)
        expect(LTN.device.type).to be == expected
      end
    end

    it "accepts a device name" do
      restore_device do
        LTN.device = "cpu"

        expect(LTN.device).to be_a(Torch::Device)
        expect(LTN.device.type).to be == "cpu"
      end
    end

    it "accepts a Torch::Device" do
      restore_device do
        device = Torch.device("cpu")
        LTN.device = device

        expect(LTN.device).to be_equal(device)
      end
    end
  end
end

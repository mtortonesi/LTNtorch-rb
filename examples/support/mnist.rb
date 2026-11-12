# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.

require "fileutils"
require "net/http"
require "zlib"
require "torch"

module LTNExamples
  # Downloads MNIST into examples/datasets/MNIST on first use (from the
  # mirror torchvision uses), and loads it as tensors: images as uint8
  # tensors of shape n x 28 x 28, labels as int64 tensors of size n. The
  # tensors are cached as a .pth file, since converting the raw bytes takes
  # a while.
  module MNIST
    MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    FILES = {
      train: %w[train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz],
      test: %w[t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz]
    }.freeze

    module_function

    def directory
      File.join(LTNExamples::DATASETS, "MNIST")
    end

    # @parameter split [Symbol] :train or :test.
    # @returns [Array(Torch::Tensor)] the images and the labels.
    def load(split)
      cache = File.join(directory, "#{split}.pth")
      if File.exist?(cache)
        tensors = Torch.load(cache)
        return [tensors["images"], tensors["labels"]]
      end

      images_file, labels_file = FILES.fetch(split).map { |name| download(name) }
      images = idx_tensor(images_file)
      labels = idx_tensor(labels_file).long
      Torch.save({"images" => images, "labels" => labels}, cache)
      [images, labels]
    end

    def download(name)
      path = File.join(directory, name)
      return path if File.exist?(path)

      FileUtils.mkdir_p(directory)
      warn "Downloading #{MIRROR}#{name}"
      response = Net::HTTP.get_response(URI("#{MIRROR}#{name}"))
      raise "Could not download #{name}: HTTP #{response.code}" unless response.is_a?(Net::HTTPSuccess)

      File.binwrite(path, response.body)
      path
    end

    # Parses an IDX file of unsigned bytes (the format of MNIST's files).
    def idx_tensor(path)
      bytes = Zlib::GzipReader.open(path, &:read)
      magic, = bytes.unpack("N")
      raise "#{path}: not an IDX file of unsigned bytes" unless (magic >> 8) == 0x08

      dimensions = magic & 0xff
      shape = bytes.unpack("N#{dimensions}", offset: 4)
      data = bytes.byteslice(4 + 4 * dimensions, shape.inject(:*))
      # Converted in chunks, to bound the size of the intermediate Ruby arrays.
      row = shape.drop(1).inject(1, :*)
      chunks = (0...shape[0]).step(1000).map do |start|
        count = [1000, shape[0] - start].min
        Torch.tensor(data.byteslice(start * row, count * row).unpack("C*"), dtype: :uint8).reshape([count] + shape.drop(1))
      end
      Torch.cat(chunks)
    end
  end
end

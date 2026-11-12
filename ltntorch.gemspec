# frozen_string_literal: true

require_relative "lib/ltn/version"

Gem::Specification.new do |spec|
  spec.name = "ltntorch"
  spec.version = LTN::VERSION

  spec.summary = "Logic Tensor Networks (LTN) for Ruby, on top of torch.rb"
  spec.description = "A Ruby port of LTNtorch, the PyTorch implementation of Logic Tensor Networks: " \
    "a neuro-symbolic framework that learns neural networks by maximizing the satisfaction " \
    "of a first-order fuzzy logic knowledge base."
  spec.authors = ["Mauro Tortonesi"]
  spec.license = "MIT"

  spec.homepage = "https://github.com/mtortonesi/LTNtorch-rb"

  spec.metadata = {
    "homepage_uri" => "https://github.com/mtortonesi/LTNtorch-rb",
    "source_code_uri" => "https://github.com/mtortonesi/LTNtorch-rb"
  }

  spec.files = Dir["{examples,lib,sig}/**/*", "*.md", base: __dir__]

  spec.required_ruby_version = ">= 3.0"

  spec.add_dependency "torch-rb", ">= 0.27.0"
end

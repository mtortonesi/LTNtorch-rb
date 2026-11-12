# frozen_string_literal: true

source "https://rubygems.org"

# Specify your gem's dependencies in ltntorch.gemspec
gemspec

# Development pin: torch-rb compiles against exactly one LibTorch version,
# and CI downloads the LibTorch matching this series (LIBTORCH_VERSION in
# .github/workflows/ci.yml). Without it, every torch-rb release breaks CI.
# The gemspec's own requirement stays open-ended for library users.
gem "torch-rb", "~> 0.27.0"

group :test do
  gem "covered", ">= 0.30.0"
  gem "sus", ">= 0.37.2"
end

group :development do
  gem "bake", ">= 0.25.0"
  gem "bake-gem", ">= 0.14.0"
  gem "bake-modernize", ">= 0.57.1"
  gem "ostruct", ">= 0.6.3" # no longer included in Ruby, so we need to explicitly add it as a dependency; required by standard
  gem "tsort", ">= 0.2.0" # no longer included in Ruby, so we need to explicitly add it as a dependency; required by standard
  gem "standard", ">= 1.56"
end

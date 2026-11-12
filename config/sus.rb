# frozen_string_literal: true

# sus loads this file's top level as a module body and prepends it onto
# Sus::Config (see Sus::Config.load), so a top-level `include` here is how
# the covered gem's documented sus integration actually wires coverage
# tracking into the test run -- not a stray mixin.
require "covered/sus"
include Covered::Sus # standard:disable Style/MixinUsage

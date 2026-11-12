# Tech stack

The present project does not use mainstream Ruby gems. Instead it uses:

- the [sus](https://github.com/socketry/sus) gem for testing (instead of, e.g., Rspec)
- the [bake](https://github.com/ioquatix/bake) gem for task management (instead of, e.g., rake)
- the [standard](https://github.com/standardrb/standard) gem for linting/formatting
- the [covered](https://github.com/socketry/covered) gem for coverage (instead of, e.g., SimpleCov) -- wired into `sus` via `config/sus.rb`; run with `COVERAGE=BriefSummary bundle exec sus` locally, or `bundle exec bake covered:validate --minimum <ratio>` to check the persisted `.covered.db` against a threshold (what CI does)
- a gems.rb file instead of a Gemfile

# Porting notes

This library is a port of [LTNtorch](https://github.com/tommasocarraro/LTNtorch) to [torch.rb](https://github.com/ankane/torch.rb). Unless a difference is documented, behavior (including numerical results) should match LTNtorch's: when in doubt, check the Python source.

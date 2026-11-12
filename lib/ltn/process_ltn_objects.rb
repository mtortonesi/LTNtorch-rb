# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2026, by Mauro Tortonesi.
# Ported from LTNtorch, copyright 2021-2024 by Tommaso Carraro.

module LTN
  # Makes the groundings of LTN objects compatible for an element-wise
  # predicate, function or connective (LTN broadcasting).
  #
  # Every object is expanded to the union of the free variables of all the
  # objects, in the order in which they first appear, by repeating its values
  # along the variables it lacks. The variable axes are then flattened into a
  # single leading batch axis, so that the i-th row of every processed object
  # refers to the same combination of individuals.
  #
  # Unlike LTNtorch, this never modifies the given objects: LTNtorch rebinds
  # the value and free variables of intermediate results (objects whose value
  # has a grad_fn), so reusing such a formula afterwards sees the broadcast
  # grounding. The processed values are the same.
  #
  # @parameter objects [Array(LTNObject)] the objects to process.
  # @returns [Array] a triple: the processed objects, the labels of all the
  #   free variables, and the number of individuals of each of them.
  # @raises [TypeError] if `objects` is not an array of LTN objects.
  def self.process_ltn_objects(objects)
    unless objects.is_a?(Array) && objects.all?(LTNObject)
      raise TypeError, "The objects should be an Array of LTNObject"
    end

    vars_to_n = {}
    objects.each do |object|
      object.free_vars.each_with_index { |var, index| vars_to_n[var] = object.shape[index] }
    end
    vars = vars_to_n.keys
    n_individuals_per_var = vars_to_n.values

    processed = objects.map do |object|
      value = object.value
      vars_in_object = object.free_vars.dup

      (vars - vars_in_object).each do |new_var|
        new_var_index = vars_in_object.size
        value = value.unsqueeze(new_var_index)
        value = Torch.repeat_interleave(value, vars_to_n[new_var], dim: new_var_index)
        vars_in_object << new_var
      end

      permutation = vars.map { |var| vars_in_object.index(var) } + (vars_in_object.size...value.dim).to_a
      value = value.permute(permutation)
      value = value.reshape([-1] + value.shape[vars_in_object.size..])

      LTNObject.new(value, vars.dup)
    end

    [processed, vars, n_individuals_per_var]
  end
end

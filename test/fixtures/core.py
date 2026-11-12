# Generates test/fixtures/core.json: reference results of LTNtorch's core
# (broadcasting, predicates, functions, connectives, quantifiers, guarded and
# diagonal quantification, and the gradients of a small knowledge base),
# which test/ltn/core_parity.rb checks the Ruby port against.
#
# Usage (needs PyTorch and a checkout of https://github.com/tommasocarraro/LTNtorch):
#
#   python test/fixtures/core.py /path/to/LTNtorch > test/fixtures/core.json
#
# Each scenario is built the same way in test/ltn/core_parity.rb, from the
# input data stored in the fixture, so the fixture does not depend on
# PyTorch's RNG.

import copy
import json
import math
import sys
import warnings

# LTNtorch's docstrings contain LaTeX escapes that recent Pythons warn about.
warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402
import ltn  # noqa: E402
import ltn.fuzzy_ops as F  # noqa: E402
from ltn import LTNObject, Constant, Variable, Predicate, Function, Connective, Quantifier, diag  # noqa: E402

ltn.device = torch.device("cpu")
torch.manual_seed(2020)


DATA = {
    "x": [[0.9, 0.4, 0.2], [-0.5, 0.3, 0.1]],                  # row sums 1.5, -0.1
    "y": [[0.3, 0.2], [-0.1, -0.1], [-0.6, 0.2], [0.5, 0.3]],  # row sums 0.5, -0.2, -0.4, 0.8
    "z": [0.25, -1.0, 0.5],                                     # 1-D: 3 individuals of 1 feature
    "c": [0.1, -0.2, 0.3, 0.4, -0.5],
    "w": [[0.2, 0.7, 0.1], [0.4, 0.1, 0.9]],                   # same individuals as x, for diag
    "t2": [[0.1, 0.5, 0.9, 0.3], [0.7, 0.2, 0.4, 0.6]],        # truth values over [x, y]
    "t3": [[0.8, 0.3, 0.6], [0.1, 0.9, 0.5], [0.4, 0.2, 0.7], [0.3, 0.6, 0.2]],  # truth values over [y, z]
    "points": [[0.1, 0.2], [0.15, 0.22], [0.8, 0.9], [0.5, 0.5], [0.52, 0.47], [0.9, 0.1]],
    "points2": [[0.12, 0.25], [0.7, 0.85], [0.5, 0.6], [0.2, 0.9], [0.55, 0.45], [0.88, 0.12]],
    "d": [0.01, 0.05, 0.3, 1.5],
}

# A small trainable model: a linear predicate over the concatenation of
# its two arguments (3 + 2 features), and a trainable constant.
linear = torch.nn.Linear(5, 1)
DATA["linear_weight"] = (torch.round(linear.weight.detach() * 1e6) / 1e6).tolist()
DATA["linear_bias"] = (torch.round(linear.bias.detach() * 1e6) / 1e6).tolist()
DATA["k"] = [0.3, -0.7]


def T(name):
    return torch.tensor(DATA[name])


def encode(value):
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def tensor_data(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "values": encode(tensor.detach().tolist()),
    }


def obj_data(obj):
    return {"value": tensor_data(obj.value), "free_vars": list(obj.free_vars)}


# Groundings shared with the Ruby side (see core_parity.rb).
P1 = lambda a: torch.sigmoid(torch.sum(a, dim=1))
P2 = lambda a, b: torch.sigmoid(torch.sum(torch.cat([a, b], dim=1), dim=1))
P2_FLAT = lambda a, b: torch.sigmoid(torch.sum(torch.cat([a, torch.flatten(b, start_dim=1)], dim=1), dim=1))
EQ = lambda a, b: torch.exp(-torch.norm(a - b, dim=1))
F2 = lambda a, b: torch.cat([a, b], dim=1)
F_MEAN = lambda a: torch.mean(a, dim=1)
F_FLAT = lambda a, b: torch.cat([a, torch.flatten(b, start_dim=1)], dim=1)

forall = Quantifier(F.AggregPMeanError(p=2), "f")
exists = Quantifier(F.AggregPMean(p=2), "e")
forall_min = Quantifier(F.AggregMin(), "f")
exists_mean = Quantifier(F.AggregMean(), "e")
and_prod = Connective(F.AndProd())
implies = Connective(F.ImpliesReichenbach())
not_ = Connective(F.NotStandard())


def variables():
    return (Variable("x", T("x")), Variable("y", T("y")), Variable("z", T("z")), Constant(T("c")))


def process(objects):
    processed, vars_, n = ltn.process_ltn_objects(objects)
    return {"objects": [obj_data(o) for o in processed], "vars": vars_, "n_individuals": n}


def process_cases():
    x, y, z, c = variables()
    xy = LTNObject(T("t2"), ["x", "y"])
    yz = LTNObject(T("t3"), ["y", "z"])
    extra = LTNObject(torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3), ["x", "y"])
    return {
        "const_const": process([c, Constant(T("x"))]),
        "const_var": process([c, x]),
        "var_var": process([x, y]),
        "three_vars": process([y, x, z]),
        "shared_var": process([xy, yz]),
        "extra_dims": process([extra, z]),
        "diag": process(list(diag(Variable("x", T("x")), Variable("w", T("w"))))),
    }


def predicate_cases():
    x, y, z, c = variables()
    p1, p2, p2_flat = Predicate(func=P1), Predicate(func=P2), Predicate(func=P2_FLAT)
    f2, f_mean, f_flat = Function(func=F2), Function(func=F_MEAN), Function(func=F_FLAT)
    return {
        "p1_x": obj_data(p1(x)),
        "p1_z": obj_data(p1(z)),
        "p2_x_y": obj_data(p2(x, y)),
        "p2_y_x": obj_data(p2(y, x)),
        "p2_x_c": obj_data(p2_flat(x, c)),
        "p1_c": obj_data(Predicate(func=lambda a: torch.sigmoid(torch.sum(a)))(c)),
        "f2_x_y": obj_data(f2(x, y)),
        "f_mean_x": obj_data(f_mean(x)),
        "f_flat_x_c": obj_data(f_flat(x, c)),
        "p2_f2": obj_data(p1(f2(x, y))),
    }


def connective_cases():
    x, y, z, c = variables()
    p1, p2 = Predicate(func=P1), Predicate(func=P2)
    xy = LTNObject(T("t2"), ["x", "y"])
    yz = LTNObject(T("t3"), ["y", "z"])
    return {
        "and_xy_yz": obj_data(and_prod(xy, yz)),
        "and_yz_xy": obj_data(and_prod(yz, xy)),
        "implies_p1x_p2xy": obj_data(implies(p1(x), p2(x, y))),
        "not_p1z": obj_data(not_(p1(z))),
        "nested": obj_data(and_prod(implies(p1(x), p2(x, y)), not_(p1(z)))),
    }


def quantifier_cases():
    x, y, z, c = variables()
    p1, p2, eq = Predicate(func=P1), Predicate(func=P2), Predicate(func=EQ)
    xy = LTNObject(T("t2"), ["x", "y"])
    yz = LTNObject(T("t3"), ["y", "z"])
    sum_x_gt = lambda a: torch.sum(a.value, dim=1) > 1.0
    sum_xy = lambda a, b: torch.logical_and(torch.sum(a.value, dim=1) > 1.0, torch.sum(b.value, dim=1) > 0.0)
    cases = {
        "forall_x": obj_data(forall(x, p2(x, y))),
        "forall_y": obj_data(forall(y, p2(x, y))),
        "forall_xy": obj_data(forall([x, y], p2(x, y))),
        "forall_x_exists_y": obj_data(forall(x, exists(y, p2(x, y)))),
        "exists_y_forall_x": obj_data(exists(y, forall(x, p2(x, y)))),
        "min_mean": obj_data(forall_min(y, exists_mean(z, yz))),
        "forall_y_of_xy_yz": obj_data(forall(y, and_prod(xy, yz))),
        "kwargs_p": obj_data(forall(x, p2(x, y), p=4)),
        "guard_x": obj_data(forall(x, p2(x, y), x, sum_x_gt)),
        "guard_xy": obj_data(forall(x, p2(x, y), [x, y], sum_xy)),
        "guard_xy_exists": obj_data(exists(x, p2(x, y), [x, y], sum_xy)),
        "guard_both": obj_data(forall([x, y], p2(x, y), [x, y], sum_xy)),
        "guard_x_over_xy": obj_data(forall([x, y], p2(x, y), x, sum_x_gt)),
        "guard_y_over_x": obj_data(forall(y, p2(x, y), x, sum_x_gt)),
        "guard_not_in_formula": obj_data(forall(x, p1(x), [x, y], sum_xy)),
    }

    v1, v2 = Variable("v1", T("x")), Variable("v2", T("w"))
    cases["diag"] = obj_data(forall(diag(v1, v2), p2(v1, v2)))
    cases["diag_reset"] = {"v1": v1.free_vars, "v2": v2.free_vars}
    near = lambda a, b: torch.abs(torch.sum(a.value, dim=1) - torch.sum(b.value, dim=1)) < 0.5
    cases["diag_guard"] = obj_data(forall(diag(v1, v2), p2(v1, v2), [v1, v2], near))

    xp, yp, d = Variable("x", T("points")), Variable("y", T("points")), Variable("d", T("d"))
    dist = lambda a, b: torch.unsqueeze(torch.norm(a.value - b.value, dim=1), 1)
    cases["eq_d"] = obj_data(exists(d, forall([xp, yp], eq(xp, yp), cond_vars=[xp, yp, d],
                                              cond_fn=lambda a, b, dd: dist(a, b) < dd.value)))
    xp, yp = Variable("x", T("points")), Variable("y", T("points2"))
    cases["eq_d_diag"] = obj_data(exists(d, forall(diag(xp, yp), eq(xp, yp), cond_vars=[xp, yp, d],
                                                   cond_fn=lambda a, b, dd: dist(a, b) < dd.value)))
    return cases


def mask_cases():
    x, y, z, c = variables()
    xy = LTNObject(T("t2"), ["x", "y"])
    sum_lt = lambda a: torch.sum(a.value, dim=1) < 0.3
    cases = {}
    for name, cond_vars, fn, agg in [
        ("y_over_x", [y], sum_lt, ["x"]),
        ("xy_over_xy", [x, y], lambda a, b: torch.sum(a.value, dim=1) < torch.sum(b.value, dim=1), ["x", "y"]),
        ("y_over_xy", [y], sum_lt, ["x", "y"]),
        ("z_not_in_formula", [z], lambda a: torch.sum(a.value, dim=1) > 0.0, ["x"]),
    ]:
        f, m = Quantifier.compute_mask(copy.deepcopy(xy), cond_vars, fn, agg)
        cases[name] = {"formula": obj_data(f), "mask": obj_data(m)}
    return cases


def knowledge_base_case():
    """Gradients of 1 - SatAgg(...) w.r.t. a linear predicate and a trainable constant."""
    x, y, z, c = variables()
    model = torch.nn.Linear(5, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(DATA["linear_weight"]))
        model.bias.copy_(torch.tensor(DATA["linear_bias"]))
    a = Predicate(func=lambda u, v: torch.sigmoid(model(torch.cat([u, v], dim=1))).squeeze(1))
    k = Constant(T("k"), trainable=True)
    p1 = Predicate(func=P1)
    sat_agg = F.SatAgg()
    axioms = [
        forall([x, y], a(x, y)),
        forall(x, a(x, k)),
        forall(x, exists(y, implies(p1(x), a(x, y)))),
        forall(x, not_(a(x, k)), x, lambda u: torch.sum(u.value, dim=1) > 0.0),
    ]
    sat = sat_agg(*axioms)
    loss = 1.0 - sat
    loss.backward()
    return {
        "axioms": [tensor_data(o.value) for o in axioms],
        "sat": tensor_data(sat),
        "loss": tensor_data(loss),
        "grad_weight": tensor_data(model.weight.grad),
        "grad_bias": tensor_data(model.bias.grad),
        "grad_k": tensor_data(k.value.grad),
    }


json.dump({
    "torch_version": torch.__version__,
    "data": DATA,
    "process": process_cases(),
    "predicates": predicate_cases(),
    "connectives": connective_cases(),
    "quantifiers": quantifier_cases(),
    "masks": mask_cases(),
    "knowledge_base": knowledge_base_case(),
}, sys.stdout, allow_nan=False)
print()

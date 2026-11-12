# Generates test/fixtures/fuzzy_ops.json: reference outputs and gradients of
# LTNtorch's fuzzy operators, which test/ltn/fuzzy_ops.rb checks the Ruby
# port against.
#
# Usage (needs PyTorch and a checkout of https://github.com/tommasocarraro/LTNtorch):
#
#   python test/fixtures/fuzzy_ops.py /path/to/LTNtorch > test/fixtures/fuzzy_ops.json
#
# Inputs are literal values rather than random draws, so the fixture does not
# depend on PyTorch's RNG, and include the edge truth values 0 and 1 where the
# stable and unstable variants of the operators differ.

import json
import math
import sys
import warnings

# LTNtorch's docstrings contain LaTeX escapes that recent Pythons warn about.
warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402
import ltn.fuzzy_ops as F  # noqa: E402

X = [[0.0, 0.1, 0.25, 0.5], [0.75, 0.9, 1.0, 0.3], [1.0, 0.0, 0.6, 0.45]]
Y = [[0.0, 0.7, 0.25, 0.2], [1.0, 0.4, 0.0, 0.3], [0.5, 1.0, 0.6, 0.05]]

# 2 x 3 x 4 groundings for the aggregators.
XS = [
    [[0.0, 0.1, 0.25, 0.5], [0.75, 0.9, 1.0, 0.3], [1.0, 0.0, 0.6, 0.45]],
    [[0.2, 0.8, 0.35, 0.65], [0.05, 0.95, 0.5, 0.15], [0.4, 0.7, 0.85, 1.0]],
]
# A mask with an all-false slice along dim 2 (row [1][1]), where masked
# means divide by zero and give NaN.
MASK = [
    [[True, False, True, True], [True, True, False, True], [False, True, True, True]],
    [[True, True, True, False], [False, False, False, False], [True, False, True, True]],
]


def encode(value):
    """Nested lists of floats, with NaN/Infinity as strings (strict JSON)."""
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
        # An input outside the autograd graph, e.g. x in ImpliesGodel(x, y),
        # which only enters through a comparison.
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "values": encode(tensor.detach().tolist()),
    }


def run(op, inputs, **kwargs):
    tensors = [torch.tensor(i, requires_grad=True) for i in inputs]
    out = op(*tensors, **kwargs)
    if not out.requires_grad:
        # e.g. NotGodel, built on torch.eq: no gradient flows through it.
        return {"output": tensor_data(out), "grads": None}
    out.sum().backward()
    return {
        "output": tensor_data(out),
        "grads": [tensor_data(t.grad) for t in tensors],
    }


def connective_cases():
    cases = []
    unary = [("NotStandard", {}, F.NotStandard()), ("NotGodel", {}, F.NotGodel())]
    for name, params, op in unary:
        cases.append({"op": name, "params": params, "call": {}, "inputs": [X], **run(op, [X])})

    binary = [
        ("AndMin", {}, F.AndMin),
        ("AndProd", {"stable": True}, F.AndProd),
        ("AndProd", {"stable": False}, F.AndProd),
        ("AndLuk", {}, F.AndLuk),
        ("OrMax", {}, F.OrMax),
        ("OrProbSum", {"stable": True}, F.OrProbSum),
        ("OrProbSum", {"stable": False}, F.OrProbSum),
        ("OrLuk", {}, F.OrLuk),
        ("ImpliesKleeneDienes", {}, F.ImpliesKleeneDienes),
        ("ImpliesGodel", {}, F.ImpliesGodel),
        ("ImpliesReichenbach", {"stable": True}, F.ImpliesReichenbach),
        ("ImpliesReichenbach", {"stable": False}, F.ImpliesReichenbach),
        ("ImpliesGoguen", {"stable": True}, F.ImpliesGoguen),
        ("ImpliesGoguen", {"stable": False}, F.ImpliesGoguen),
        ("ImpliesLuk", {}, F.ImpliesLuk),
    ]
    for name, params, klass in binary:
        op = klass(**params)
        cases.append({"op": name, "params": params, "call": {}, "inputs": [X, Y], **run(op, [X, Y])})

    # The per-call `stable` keyword overrides the constructor's.
    for name, klass in [("AndProd", F.AndProd), ("OrProbSum", F.OrProbSum),
                        ("ImpliesReichenbach", F.ImpliesReichenbach), ("ImpliesGoguen", F.ImpliesGoguen)]:
        op = klass(stable=True)
        cases.append({"op": name, "params": {"stable": True}, "call": {"stable": False},
                      "inputs": [X, Y], **run(op, [X, Y], stable=False)})

    # Broadcasting between operands of different shapes.
    op = F.AndProd()
    cases.append({"op": "AndProd", "params": {"stable": True}, "call": {},
                  "inputs": [X, Y[0]], **run(op, [X, Y[0]])})

    for and_name, and_cls in [("AndProd", F.AndProd), ("AndMin", F.AndMin)]:
        for imp_name, imp_cls in [("ImpliesReichenbach", F.ImpliesReichenbach), ("ImpliesGodel", F.ImpliesGodel)]:
            op = F.Equiv(and_cls(), imp_cls())
            cases.append({"op": "Equiv", "params": {"and_op": and_name, "implies_op": imp_name}, "call": {},
                          "inputs": [X, Y], **run(op, [X, Y])})
    return cases


def aggregator_cases():
    cases = []
    dims = [None, [0], [2], [1, 2], [0, 1, 2]]
    ops = [("AggregMin", {}, F.AggregMin), ("AggregMean", {}, F.AggregMean)]
    for p in [1, 2, 3]:
        for stable in [True, False]:
            ops.append(("AggregPMean", {"p": p, "stable": stable}, F.AggregPMean))
            ops.append(("AggregPMeanError", {"p": p, "stable": stable}, F.AggregPMeanError))

    for name, params, klass in ops:
        op = klass(**params)
        for dim in dims:
            for keepdim in [False, True]:
                call = {"dim": dim, "keepdim": keepdim}
                kwargs = {"dim": tuple(dim) if dim is not None else None, "keepdim": keepdim}
                cases.append({"op": name, "params": params, "call": call, "inputs": [XS], **run(op, [XS], **kwargs)})
        for dim in [[2], [1, 2]]:
            call = {"dim": dim, "keepdim": False, "mask": MASK}
            mask = torch.tensor(MASK)
            cases.append({"op": name, "params": params, "call": call, "inputs": [XS],
                          **run(op, [XS], dim=tuple(dim), mask=mask)})

    # Per-call p and stable override the constructor's.
    for name, klass in [("AggregPMean", F.AggregPMean), ("AggregPMeanError", F.AggregPMeanError)]:
        op = klass(p=2, stable=True)
        call = {"dim": [2], "keepdim": False, "p": 4, "stable": False}
        cases.append({"op": name, "params": {"p": 2, "stable": True}, "call": call, "inputs": [XS],
                      **run(op, [XS], dim=(2,), p=4, stable=False)})
    return cases


def sat_agg_cases():
    cases = []
    truth_values = [0.2, 0.9, 1.0, 0.0, 0.65]
    for name, params, agg in [
        ("AggregPMeanError", {"p": 2, "stable": True}, F.AggregPMeanError(p=2)),
        ("AggregMin", {}, F.AggregMin()),
        ("AggregPMean", {"p": 3, "stable": True}, F.AggregPMean(p=3)),
    ]:
        tensors = [torch.tensor(v, requires_grad=True) for v in truth_values]
        out = F.SatAgg(agg)(*tensors)
        out.backward()
        cases.append({
            "agg_op": name, "params": params, "inputs": truth_values,
            "output": tensor_data(out), "grads": [tensor_data(t.grad) for t in tensors],
        })
    return cases


json.dump({
    "torch_version": torch.__version__,
    "connectives": connective_cases(),
    "aggregators": aggregator_cases(),
    "sat_agg": sat_agg_cases(),
}, sys.stdout, allow_nan=False)
print()

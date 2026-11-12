# Generates test/fixtures/training.json: the loss, at each epoch, of training
# a small knowledge base with LTNtorch, from fixed data and initial weights,
# which test/ltn/training_parity.rb reproduces with the Ruby port.
#
# Usage (needs PyTorch and a checkout of https://github.com/tommasocarraro/LTNtorch):
#
#   python test/fixtures/training.py /path/to/LTNtorch > test/fixtures/training.json
#
# The knowledge base is that of LTNtorch's binary classification example,
# plus a guarded axiom, so that float64 truth values take part in training.

import json
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402
import ltn  # noqa: E402

ltn.device = torch.device("cpu")
torch.manual_seed(2020)

EPOCHS = 60

data = torch.round(torch.rand((50, 2)) * 1e6) / 1e6
labels = torch.sum(torch.square(data - torch.tensor([.5, .5])), dim=1) < .09


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 16)
        self.layer2 = torch.nn.Linear(16, 16)
        self.layer3 = torch.nn.Linear(16, 1)
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.elu(self.layer1(x))
        x = self.elu(self.layer2(x))
        return self.sigmoid(self.layer3(x))


model = Model()
with torch.no_grad():
    for parameter in model.parameters():
        parameter.copy_(torch.round(parameter * 1e6) / 1e6)
initial = {name: value.tolist() for name, value in model.state_dict().items()}

A = ltn.Predicate(model)
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()
optimizer = torch.optim.Adam(A.parameters(), lr=0.01)

losses = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    x_A = ltn.Variable("x_A", data[labels])
    x_not_A = ltn.Variable("x_not_A", data[torch.logical_not(labels)])
    x = ltn.Variable("x", data)
    sat = SatAgg(
        Forall(x_A, A(x_A)),
        Forall(x_not_A, Not(A(x_not_A))),
        # points far from the center are not A
        Forall(x, Not(A(x)), cond_vars=[x],
               cond_fn=lambda x: torch.sum(torch.square(x.value - 0.5), dim=1) > 0.2),
    )
    loss = 1. - sat
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

json.dump({
    "torch_version": torch.__version__,
    "data": data.tolist(),
    "labels": labels.tolist(),
    "initial": initial,
    "losses": losses,
}, sys.stdout)
print()

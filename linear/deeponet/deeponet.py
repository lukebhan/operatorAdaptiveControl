import torch

class DeepONet(torch.nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()

        self.branch = branch_net
        self.trunk = trunk_net
        self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, values, grid):
        branchRes = self.branch(values)
        trunkRes = self.trunk(grid)
        sumres = torch.einsum('abc,abc->ab', branchRes, trunkRes)
        sumres += self.b
        return sumres




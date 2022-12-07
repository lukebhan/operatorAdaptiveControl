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
 
class DeepONet2D(torch.nn.Module):

    def __init__(self, branch_net, trunk_net, proj):
        super().__init__()
        self.proj = proj
        self.branch = branch_net
        self.trunk = trunk_net
        self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, values, grid):
        branchRes = self.branch(values)
        trunkRes = self.trunk(grid)
        bSize = branchRes.size()
        tSize = trunkRes.size()
        branchRes = torch.reshape(branchRes, (bSize[0], bSize[1], self.proj, bSize[2] // self.proj))
        trunkRes = torch.reshape(trunkRes, (tSize[0], tSize[1], self.proj, tSize[2] // self.proj))
        sumres = torch.einsum('abcd,abcd->abc', branchRes, trunkRes)
        sumres += self.b
        return sumres




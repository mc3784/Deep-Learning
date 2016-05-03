
require 'torch'
require 'nngraph'
local input=torch.Tensor(2):fill(0)
local prevh=torch.Tensor(2):fill(1)
nhid=2

local i2h = nn.Linear(nhid, 3 * nhid)(input)
local h2h = nn.Linear(nhid, 3 * nhid)(prevh)
local gates = nn.CAddTable()({
nn.Narrow(2, 1, 2 * nhid)(i2h),
nn.Narrow(2, 1, 2 * nhid)(h2h),
})
gates = nn.SplitTable(2)(nn.Reshape(2, nhid)(gates))
local resetgate = nn.Sigmoid()(nn.SelectTable(1)(gates))
local updategate = nn.Sigmoid()(nn.SelectTable(2)(gates))
local output = nn.Tanh()(nn.CAddTable()({
nn.Narrow(2, 2 * nhid+1, nhid)(i2h),
nn.CMulTable()({resetgate,
nn.Narrow(2, 2 * nhid+1, nhid)(h2h),})
}))
local nexth = nn.CAddTable()({ prevh,
nn.CMulTable()({ updategate,
nn.CSubTable()({output, prevh,}),}),
})
mlp = nn.gModule({input}, {nexth})
graph.dot(mlp.fg, 'MLP')

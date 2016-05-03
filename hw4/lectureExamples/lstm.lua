function lstmcell(input, prevh)
local i2h = nn.Linear(nhid, 4 * nhid)(input)
local h2h = nn.Linear(nhid, 4 * nhid)(prevh)
local gates = nn.CAddTable()({i2h, h2h})
gates = nn.Reshape(4,nhid)(gates)
gates = nn.SplitTable(2)(gates)
local ingate = nn.Sigmoid()(nn.SelectTable(1)(gates))
local intransform = nn.Tanh()( nn.SelectTable(2)(gates))
local forgetgate = nn.Sigmoid()(nn.SelectTable(3)(gates))
local outgate = nn.Sigmoid()(nn.SelectTable(4)(gates))
local nextc = nn.CAddTable()({
nn.CMulTable()({forgetgate, prevc}),
nn.CMulTable()({ingate, intransform})
})
local nexth = nn.CMulTable()({outgate, nn.Tanh()(nextc)})
return {nextc, nexth}
end

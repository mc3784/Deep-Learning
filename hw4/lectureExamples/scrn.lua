require ‘nngraph’
local a = 0.9
local nhid = 100
local nstr = 40
local input = nn.Identity()()
local prevh = nn.Identity()()
local prevs = nn.Identity()()
local h2h = nn.Linear(nhid, nhid, false)(prevh)
local s2h = nn.Linear(nstr, nhid, false)(prevs)
local i2h = nn.LookupTable(ninput, nhid)(input)
local i2s = nn.LookupTable(ninput, nstr)(input)
local h = nn.Sigmoid()(nn.CAddTable()({i2h, h2h, s2h})
local s = nn.CAddTable(){nn.Mul(1-a)(i2s), nn.Mul(a)(prevs)}
local y = nn.Linear(nhid + nstr, nvoc, false)(nn.ConcatTable{h,s})
scrn = nn.gModule({prevh, prevs, input}, {h, s, y})

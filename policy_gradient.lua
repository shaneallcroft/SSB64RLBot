local _ = require 'moses'
local nn = require 'nn'
local gnuplot = require 'gnuplot'
local environ = require 'environ'
local socket = require 'socket'
local string = require 'string'
local newScore = -1
local s = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
--local os = require 'os'

local server = assert(socket.bind("*", 51111))
local ip, port = server:getsockname()
print("ip:" .. ip)
print("port:" .. port)

function split(pString, pPattern)
  local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
  local fpat = "(.-)" .. pPattern
  local last_end = 1
  local s, e, cap = pString:find(fpat, 1)
  while s do
     if s ~= 1 or cap ~= "" then
    table.insert(Table,cap)
     end
     last_end = e+1
     s, e, cap = pString:find(fpat, last_end)
  end
  if last_end <= #pString then
     cap = pString:sub(last_end)
     table.insert(Table, cap)
  end
  return Table
end

local client = server:accept() --wait for the client to connect

-- Load Q* from MC control
local QStar = torch.load('Q.t7')
local V = torch.max(QStar, 3):squeeze()

-- Set manual seed
torch.manualSeed(1)

local nEpisodes = 100
-- Number of discrete actions
local m = #environ.A

-- Initial exploration ɛ
local epsilon = 1
-- Linear ɛ decay
local epsilonDecay = 1/nEpisodes
-- Minimum ɛ
local epsilonMin = 0.05
-- Constant step-size ɑ
local alpha = 0.001
-- RMSProp decay
local decay = 0.9
-- (Machine) epsilon
local eps = 1e-20
-- Entropy regularisation factor β
local beta = 0.01

--PATH = "smashModel.pt"					OS THING
local net = nil
--if os.path.isdir(PATH) then					OS THING
--  net = torch.load(PATH)					OS THING
--else					OS THING

  -- Create policy network π
  net = nn.Sequential()
  input = 15
  net:add(nn.Linear(input, 16))
  net:add(nn.ReLU(true))
  net:add(nn.Linear(16, m))
  net:add(nn.SoftMax())
--end					OS THING

-- Get network parameters θ
local theta, gradTheta = net:getParameters()
-- Moving average of squared gradient
local gradThetaSq = torch.Tensor(gradTheta:size()):zero()

-- Results from each episode
local results = torch.Tensor(nEpisodes)

-- Sample
for i = 1, nEpisodes do
  --TODO: Start new game and load the new game's first state accordingly 
  local reception = nil

	while (reception == nil) do
		client:settimeout(1)
		reception = client:receive()
		--print (reception)
	end
		  --print("FIRST I RECIEVED" .. reception)

  -- Experience tuples (s, a, r)
  local E = {}
  -- {bot death state, bot damage taken, bot x pos, bot y, bot xvel, bot yvel, } 

  s = split(reception, ",")
  -- Run till termination
  newScore = 0
  repeat
    -- Choose action by ɛ-greedy exploration
    local aIndex
    if torch.uniform() < (1 - epsilon) then -- Exploit with probability 1 - ɛ
      -- Get categorical action distribution from π = p(s; θ)
      local probs = net:forward(torch.Tensor(s))
      probs:add(eps) -- Add small probability to prevent NaNs
      -- Sample action ~ p(s; θ)
      aIndex = torch.multinomial(probs, 1)[1]
    else
      -- Otherwise pick any action with probability 1/m
      aIndex = torch.random(1, m)
    end
    local a = environ.A[aIndex]
    --print(a)

    local oldS = s
    local oldScore = newScore-- before newScore actually gets updated put it in old score 
    -- Perform a step

    -- Have player perform Action

    -- Send Action to server
    client:send(a .. "\n")
    -- Wait and Recieve new state from server
    local reception2 = nil
	  while (reception2 == nil) do
		client:settimeout(1)
		reception2 = client:receive()
		--print (reception)
	  end
	   --print("I RECIEVED" .. reception2)

    s = split(reception2, ',')
    newScore = tonumber(table.remove(s, 1))
    -- Score based on how well the action performed
    local r = environ.calculateReward(newScore, oldScore) -- r comes from score function f(s)

    -- Store experience tuple
    table.insert(E,  {oldS, a, r})

    -- Linearly decay ɛ"
    epsilon = math.max(epsilon - epsilonDecay, epsilonMin)
  print(s[1] .. '\n')
  print(s[8] .. '\n')
  print(newScore)
  until environ.isTerminal(s[1], s[8])
  
  -- Save result of episode
  results[i] = E[#E][3]
  
  -- Reset ∇θ
  gradTheta:zero()
  
  -- Learn from experience of one complete episode
  for j = 1, #E do
    -- Extract experience
    local s = E[j][1]
    local a = E[j][2]
    -- Get action index
    local aIndex = _.find(environ.A, a)

    -- Calculate variance-reduced reward (advantage) ∑t r - b(s) = ∑t r - V(s) = A
    local A = 0
    for k = j, #E do
      print("pog")
      A = A + (E[k][3] - V[s[1]][s[2]])
      print("champ")
    end

    -- Use a policy gradient update (REINFORCE rule): ∇θ Es[f(s)] = ∇θ ∑s p(s)f(s) = Es[f(s) ∇θ logp(s)]
    local input = torch.Tensor(s)
    local output = net:forward(input)
    output:add(eps) -- Add small probability to prevent NaNs

    -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
    local target = torch.zeros(m)
    target[aIndex] = A * 1/output[aIndex] -- f(s) ∇θ logp(s)

    -- Calculate gradient of entropy of policy: -logp(s) - 1
    local gradEntropy = -torch.log(output) - 1
    -- Add to target to improve exploration (prevent convergence to suboptimal deterministic policy)
    target:add(beta * gradEntropy)
    
    -- Accumulate gradients
    net:backward(input, target)
  end

  -- Update moving average of squared gradients
  gradThetaSq = decay * gradThetaSq + (1 - decay) * torch.pow(gradTheta, 2)
  -- RMSProp update (gradient ascent version)
  theta:add(torch.cdiv(alpha * gradTheta, torch.sqrt(gradThetaSq) + eps))
end

-- Take average results over 1000 episodes
local avgResults = torch.Tensor(nEpisodes/1000)
for ep = 1, nEpisodes, 1000 do
  avgResults[(ep - 1)/1000 + 1] = torch.mean(results:narrow(1, ep, 1000))
end

-- Plot results
gnuplot.pngfigure('PolicyGradient.png')
gnuplot.plot('Average Result', torch.linspace(1, nEpisodes/1000, nEpisodes/1000), avgResults)
gnuplot.title('Policy Gradient Results')
gnuplot.ylabel('Result (Mean over 1000 Episodes)')
gnuplot.xlabel('Episode (x1000)')
gnuplot.plotflush()

torch.save(net, PATH)
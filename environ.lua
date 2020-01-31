local environ = {}
local math = require "math"
-- States are a tuple of dealer's first card (1-10) and the player's sum
-- Set of actions

environ.A = {'no input', 'P1 A', 'P1 B', 'P1 R', 'P1 Z', 'up full', 'up right full', 'right full', 'down right full', 'down full', 'down left full', 'left full', 'up left full', 'A + up full', 'A + right full',  'A + down full', 'A + left full', 'B + up full', 'B + down full', 'Z + left full', 'Z + right full'} --rolling
-- 21 possible actions
-- State transitions are not explicitly stored

-- Draws a card (with replacement)
local drawCard = function()
  -- Draw number uniformly from 1-10
  local num = torch.random(1, 10)
  -- Draw colour red with p = 1/3 and black with p = 2/3
  if torch.uniform() <= 1/3 then
    return {num, 'red'}
  else
    return {num, 'black'}
  end
end

-- Adds a card to an existing sum
local addCard = function(sum, card)
  if card[2] == 'black' then
    return sum + card[1]
  else
    return sum - card[1]
  end
end

-- Checks if bust
local checkBust = function(sum)
  return sum > 21 or sum < 1
end

-- Perform a step in the environment
environ.calculateReward = function(newScore, oldScore)
  -- Enemy Death * (11000) + Self Death * (-10000) + Enemy Percent * (300) + Self Percent * (-143) + Self Destruct * (-30000) + 
  local r = tonumber(newScore) - tonumber(oldScore)

  return r
end


--[[
-- Performs a step in the environment
environ.step = function(s, a)
  -- Current state
  local dealersFirstCard = s[1]
  local playersSum = s[2]
  -- Next state (initialised with deep copy of current state)
  local sPrime = {}
  sPrime[1] = dealersFirstCard
  sPrime[2] = playersSum
  -- Reward (0 by default)
  local r = 0
  -- Process actions
  if a == 'hit' then
    -- Draw and add next card
    sPrime[2] = addCard(playersSum, drawCard())
    -- Check player fail conditions
    if checkBust(sPrime[2]) then
      r = -1
    end
  elseif a == 'stick' then
    local dealersSum = dealersFirstCard
    -- Dealer sticks when on 17 or higher
    while dealersSum < 17 and dealersSum >= 1 do
      -- Draw and add next card
      dealersSum = addCard(dealersSum, drawCard())
    end
    -- Check dealer fail conditions and winning conditions otherwise (player with largest sum)
    if checkBust(dealersSum) or dealersSum < playersSum then
      r = 1
    elseif dealersSum > playersSum then
      r = -1
    end
  end
  return sPrime, r
end
]]--

-- Calculates if the current state is terminal given previous action and reward
environ.isTerminal = function(self_deaths, enemy_deaths)
  -- if self_deaths or enemy_deaths
  if tonumber(self_deaths) >= 1 or tonumber(enemy_deaths) >= 1 then
    return true
  else
    return false
  end
end

environ.cartesianDistanceFromEnemy = function()
    Xdiff = self_x - enemy_x
    Ydiff = self_y - enemy_y
    XdiffSquared = Xdiff * Xdiff
    YdiffSquared = Ydiff * Ydiff
    return math.sqrt(XdiffSquared + YdiffSquared)
end

return environ
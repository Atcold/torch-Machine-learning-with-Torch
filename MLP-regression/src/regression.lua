--------------------------------------------------------------------------------
-- Regression with Neural Networks
--------------------------------------------------------------------------------
-- Alfredo Canziani, Jul 14
--------------------------------------------------------------------------------

-- Instruction -----------------------------------------------------------------

-- + Monitor the progress
-- You can run this script with small (1e2) values for <trainer.maxIteration>
-- and see how the MLP learns the target function through <nbSnapShots> of
-- partial results. Have a look also how the cost function decreses with the
-- progress of the training. Notice that, being randomly initialised, the out-
-- put will change if the script is run another time.

-- + Check each neuron
-- You can also choose a higer (1e4) <trainer.maxIteration> and then set to
-- false <plotIntermediateResults>. In this way you will be able to check an
-- optimal regressed function along with the output of the three hidden neurones.

-- Choose a combination (comment the other)
--<<<
plotIntermediateResults = true
maxIteration = 1e2
--<<<>>>
--plotIntermediateResults = false
--maxIteration = 1e4
-->>>

-- Requires --------------------------------------------------------------------
require 'nn'
require 'gnuplot'

-- Define dataset --------------------------------------------------------------
dataset = {}
function dataset:size() return 50 end
x = torch.linspace(-1,1,dataset:size())
y = x:clone():pow(2)
for i = 1, dataset:size() do
   dataset[i] = {x:reshape(x:size(1),1)[i], y:reshape(y:size(1),1)[i]}
end

-- Define model architecture ---------------------------------------------------
model = nn.Sequential()
model:add(nn.Linear(1,3))
model:add(nn.Tanh())
model:add(nn.Linear(3,1))

-- Trainer definition ----------------------------------------------------------
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = maxIteration
trainer.verbose = false

-- Hook iteration function
nbSnapShots = 5
h = torch.Tensor(nbSnapShots+1, x:size(1))
plt = {{'Training data', x, y, '+'}}

-- Starting condition
if plotIntermediateResults then
   for i = 1, x:size(1) do
      h[#plt][i] = model:forward(x:reshape(x:size(1),1)[i])[1]
   end
   table.insert(plt,{'Iteration ' .. 0, x, h[#plt], '-'})
end
gnuplot.plot(plt)
gnuplot.grid(true)

nbErrorPoints = 20
costFunction = {}
costIter = {}

function trainer.hookIteration(train, iteration, currentError)
   if iteration % (train.maxIteration/nbSnapShots) == 0 then
      if plotIntermediateResults then
         for i = 1, x:size(1) do
            h[#plt][i] = model:forward(x:reshape(x:size(1),1))[i]
         end
         table.insert(plt,{'Iteration ' .. iteration, x, h[#plt], '-'})
         gnuplot.figure(1)
         gnuplot.plot(plt)
      end
      print("# Epoch " .. iteration .. ", error: ", currentError)
   end
   if iteration % (train.maxIteration/nbErrorPoints) == 0 or iteration == 1 then
      table.insert(costFunction, currentError)
      table.insert(costIter, iteration)
      gnuplot.figure(2)
      gnuplot.plot{'Cost function', torch.Tensor(costIter), torch.Tensor(costFunction)}
      gnuplot.xlabel('Iterations')
      gnuplot.grid(true)
   end
end

-- Training --------------------------------------------------------------------
-- Start timer
timer = torch.Timer()

-- Training
trainer:train(dataset)

-- Profiling
t = timer:time().real
print('Time per iteration [ms]: ', t * 1000 / trainer.maxIteration)
print('Total training time [min]: ', t / 60)

-- Check neurones --------------------------------------------------------------
if not plotIntermediateResults then
   gnuplot.figure(1)
   for i = 1, x:size(1) do
      h[1][i] = model:forward(x:reshape(x:size(1),1)[i])[1]
      h[2][i] = model.modules[2].output[1]
      h[3][i] = model.modules[2].output[2]
      h[4][i] = model.modules[2].output[3]
   end
   table.insert(plt,{'Regression', x, h[1], '-'})
   table.insert(plt,{'Neuron 1', x, h[2], '-'})
   table.insert(plt,{'Neuron 2', x, h[3], '-'})
   table.insert(plt,{'Neuron 3', x, h[4], '-'})
   gnuplot.plot(plt)
end

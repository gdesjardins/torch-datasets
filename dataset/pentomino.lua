require 'torch'
require 'dok'
require 'paths'
require 'hdf5'

require 'util'
require 'util/arg'
local arg = util.arg

require 'dataset'
require 'dataset/TableDataset'

Pentomino = {}
Pentomino.name         = 'pentomino'
Pentomino.dimensions   = {1, 64, 64}
Pentomino.n_dimensions = 1 * 64 * 64
Pentomino.classes      = {[1] = 0, 1}

-- Setup a Pentomino dataset instance.
function Pentomino.dataset(opts)


    local file, test, numFolds, fold, center
    opts          = opts or {}
    file          = arg.optional(opts, 'file', 'pento64x64_10k_seed_23111298122_64patches.h5')
    test          = arg.optional(opts, 'test', false)
    numFolds      = arg.optional(opts, 'numFolds', 5)
    fold          = arg.optional(opts, 'fold', 1)
    center        = arg.optional(opts, 'center', false)
    assert(fold <= numFolds)
    Pentomino.file = opts.file

    local fname = paths.concat(dataset.get_data_dir(), Pentomino.name, file)
    local fp = hdf5.open(fname, 'r')
    local _data = fp:read('x'):all():double()
    local _labels = fp:read('y'):all():double():add(1)
    fp:close()

    local totSize = _data:size(1)
    local foldSize = totSize / numFolds
    local foldIndices = torch.range(1, totSize, foldSize):storage():totable()

    local testData = _data[{{foldIndices[fold], foldIndices[fold] + foldSize - 1},{}}]
    local testLabels = _labels[{{foldIndices[fold], foldIndices[fold] + foldSize - 1}}]

    table.remove(foldIndices, fold)

    local trainData, trainLabels
    for fold=1,#foldIndices do
      local dataFold = _data[{{foldIndices[fold], foldIndices[fold] + foldSize - 1},{}}]
      local labelsFold = _labels[{{foldIndices[fold], foldIndices[fold] + foldSize - 1}}]
      if fold == 1 then
          trainData = dataFold
          trainLabels = labelsFold
      else
          trainData = torch.cat(trainData, dataFold, 1)
          trainLabels = torch.cat(trainLabels, labelsFold, 1)
      end
    end

    if center then
        local trainMean = trainData:mean(1)
        trainData:add(-trainMean:expandAs(trainData))
        testData:add(-trainMean:expandAs(testData))
    end

    local d
    if test then
       Pentomino.size = testData:size(1)
       d = dataset.TableDataset({data = testData, class = testLabels}, Pentomino)
    else
       Pentomino.size = trainData:size(1)
       d = dataset.TableDataset({data = trainData, class = trainLabels}, Pentomino)
    end

   return d
end

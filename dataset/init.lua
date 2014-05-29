require 'paths'
require 'torch'
require 'image'

require 'util'
require 'util/file'

local TORCH_DIR = paths.concat(os.getenv('HOME'), '.torch')
local DATA_DIR  = paths.concat(TORCH_DIR, 'data')

dataset = {}

-- Check locally and download dataset if not found.  Returns the path to the
-- downloaded data file.
function dataset.get_data(name, url)
  local dset_dir   = paths.concat(DATA_DIR, name)
  local data_file = paths.basename(url)
  local data_path = paths.concat(dset_dir, data_file)

  print("checking for file located at: ", data_path)

  check_and_mkdir(TORCH_DIR)
  check_and_mkdir(DATA_DIR)
  check_and_mkdir(dset_dir)
  check_and_download_file(data_path, url)

  return data_path
end


-- Downloads the data if not available locally, and returns local path.
function dataset.data_path(name, url, file)
    local data_path  = dataset.get_data(name, url)
    local data_dir   = paths.dirname(data_path)
    local local_path = paths.concat(data_dir, file)

    if not is_file(local_path) then
        do_with_cwd(data_dir,
          function()
              print("decompressing file: ", data_path)
              decompress_file(data_path)
          end)
    end

    return local_path
end


function dataset.scale(data, min, max)
    local range = max - min
    local dmin = data:min()
    local dmax = data:max()
    local drange = dmax - dmin

    data:add(-dmin)
    data:mul(range)
    data:mul(1/drange)
    data:add(min)
end


function dataset.rand_between(min, max)
   return math.random() * (max - min) + min
end


function dataset.rand_pair(v_min, v_max)
   local a = dataset.rand_between(v_min, v_max)
   local b = dataset.rand_between(v_min, v_max)
   return a,b
end


function dataset.sort_by_class(samples, labels)
    local size = labels:size()[1]
    local sorted_labels, sort_indices = torch.sort(labels)
    local sorted_samples = samples.new(samples:size())

    for i=1, size do
        sorted_samples[i] = samples[sort_indices[i]]
    end

    return sorted_samples, sorted_labels
end

--[[
Given a dataset with N classes, splits a dataset into its respective classes.

Parameters:
* samples (torch.Tensor) training examples
* labels (torch.Tensor) training labels
* classes (table or torch.Tensor) class labels, i.e. {1,2,...,10} for MNIST
* table where the i-th entry contains all the rows of `samples` corresponding
        to the i-th class.
]]
function dataset.split_by_class(samples, labels, classes)
    assert(samples:size(1) == labels:size(1))
    local sorted_classes, _ = torch.sort(torch.Tensor(classes))
    local sorted_samples, sorted_labels = dataset.sort_by_class(samples, labels)
    
    local current_class = sorted_labels[1]
    local current_class_index = 1
    local split_samples = {} 
    for i=1, labels:size(1) do
       if sorted_labels[i] ~= current_class then
          table.insert(split_samples, sorted_samples[{{current_class_index, i-1}}])
          current_class = sorted_labels[i]
          current_class_index = i
       end
    end
   
    table.insert(split_samples, sorted_samples[{{current_class_index, sorted_samples:size(1)}}])
    
    return split_samples
end

local function count_classes(labels, labels_to_count)
    local nElem = 0
    for _, label_to_count in ipairs(labels_to_count) do
        nElem = nElem + labels:eq(label_to_count):sum()
    end
    return nElem
end

--[[
Build a new dataset, extracting all examples with labels in `labels_to_include`.
e.g. on MNIST, calling this function with labels_to_include={1,2,5} will create
a new dataset containing digits 1, 2 and 5 only.

Parameters:
* samples (torch.Tensor) training examples
* labels (torch.Tensor) training labels
* labels_to_include (table or torch.Tensor) class labels to include

Returns: pair of torch.Tensors containing the extracted examples and their respective labels.
]]
function dataset.include_by_class(samples, labels, labels_to_include)
    assert(samples:size(1) == labels:size(1))
    local nElem = count_classes(labels, labels_to_include)
    local samples_size = samples:size()
    samples_size[1] = nElem
    
    local class_samples = torch.Tensor(samples_size)
    local class_labels = torch.Tensor(nElem)
    local output_index = 1
    for i=1, samples:size(1) do
        for _, label_to_include in ipairs(labels_to_include) do
            if labels[i] == label_to_include then
                class_samples[{{output_index}}] = samples[{{i}}]
                class_labels[{{output_index}}] = label_to_include
                output_index = output_index + 1
            end
        end
    end

    return class_samples, class_labels
end

function dataset.rotator(start, delta)
   local angle = start
   return function(src, dst)
      image.rotate(dst, src, angle)
      angle = angle + delta
   end
end


function dataset.translator(startx, starty, dx, dy)
   local started = false
   local cx = startx
   local cy = starty
   return function(src, dst)
      image.translate(dst, src, cx, cy)
      cx = cx + dx
      cy = cy + dy
   end
end


function dataset.zoomer(start, dz)
   local factor = start
   return function(src, dst)
      local src_width  = src:size(2)
      local src_height = src:size(3)
      local width      = math.floor(src_width * factor)
      local height     = math.floor(src_height * factor)

      local res = image.scale(src, width, height)
      if factor > 1 then
         local sx = math.floor((width - src_width) / 2)+1
         local sy = math.floor((height - src_height) / 2)+1
         dst:copy(res:narrow(2, sx, src_width):narrow(3, sy, src_height))
      else
         local sx = math.floor((src_width - width) / 2)+1
         local sy = math.floor((src_height - height) / 2)+1
         dst:zero()
         dst:narrow(2, sx,  width):narrow(3, sy, height):copy(res)
      end

      factor = factor + dz
   end
end


--[[
Split a TableDataset into two, e.g. to use as train (dev) and validation set.

Parameters:
* tableDataset instance of the type dataset.TableDataset
* opts.ratio proportion of training instances to use for "new" dataset

Returns: table of length two. First entry is the original tableDataset, modified
to only have the first (1 - opts.ratio) * nElements training instances. The
second entry is a new TableDataset having the last opts.ratio * nElements.

Note: If you intend to use this function to split a training set into train/valid, you
should first ensure that the original dataset is randomized.
]]
function dataset.splitter(tableDataset, opts)
    local ratio = opts.ratio or 0.1
    local nElem = tableDataset:size()
    local nSplit = math.floor(nElem * (1 - ratio) + 0.5)
  
    -- create new TableDataset from original dataset
    local dataTable = {data = tableDataset.dataset.data[{{nSplit+1, nElem}}]}
    if tableDataset.dataset.class then
        dataTable.class = tableDataset.dataset.class[{{nSplit+1, nElem}}]
    end
    local newTableDataset = dataset.TableDataset(dataTable, tableDataset:metadata())

    -- strip away old data from original dataset
    tableDataset.dataset.data = tableDataset.dataset.data[{{1,nSplit}}] 
    if tableDataset.dataset.class then
        tableDataset.dataset.class = tableDataset.dataset.class[{{1,nSplit}}] 
    end
    
    return {tableDataset, newTableDataset}
end

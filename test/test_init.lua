require 'dataset'
require 'dataset/TableDataset'
require 'totem'

tests = {}
local tester = totem.Tester()
local precision = 1e-8

function tests.test_split_by_class1()
    local dset = {
        data  = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}}),
        class = torch.Tensor({  1,   2,   2,   1,   3,   2,   1}),
        classes = torch.Tensor({1,2,3}),
    }
    local rval = dataset.split_by_class(dset.data, dset.class, dset.classes)
    tester:assert(#rval == dset.classes:size(1), 'split_by_class returned the wrong number of classes')
    tester:assertTensorEq(rval[1], torch.Tensor({{1},{7},{4}}), precision)
    tester:assertTensorEq(rval[2], torch.Tensor({{6},{3},{2}}), precision)
    tester:assertTensorEq(rval[3], torch.Tensor({{5}}), precision)
end

function tests.test_split_by_class_singleClassOnly()
    local dset = {
        data  = torch.Tensor({{1}, {2}, {3}}),
        class = torch.Tensor({2, 2, 2}),
        classes = torch.Tensor({2}),
    }
    local rval = dataset.split_by_class(dset.data, dset.class, dset.classes)
    tester:assert(#rval == 1)
    tester:assertTensorEq(rval[1], torch.Tensor({{3},{2},{1}}), precision)
end

function tests.test_split_by_class1()
    local dset = {
        data  = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}}),
        class = torch.Tensor({  1,   2,   2,   1,   3,   2,   1}),
        classes = torch.Tensor({1,2,3}),
    }
    local rval = dataset.split_by_class(dset.data, dset.class, dset.classes)
    tester:assert(#rval == dset.classes:size(1), 'split_by_class returned the wrong number of classes')
    tester:assertTensorEq(rval[1], torch.Tensor({{1},{7},{4}}), precision)
    tester:assertTensorEq(rval[2], torch.Tensor({{6},{3},{2}}), precision)
    tester:assertTensorEq(rval[3], torch.Tensor({{5}}), precision)
end

function tests.test_include_by_class()
    local dset = {
        data  = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}}),
        class = torch.Tensor({  1,   2,   2,   1,   3,   2,   1}),
        classes = torch.Tensor({1,2,3}),
    }
    local data, labels = dataset.include_by_class(dset.data, dset.class, {1,3})
    tester:assertTensorEq(data, torch.Tensor({{1},{4},{5},{7}}), precision)
    tester:assertTensorEq(labels, torch.Tensor({1,1,3,1}), precision)
end

return tester:add(tests):run()



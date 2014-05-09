require 'fn/seq'
require 'dataset'
require 'dataset/TableDataset'
require 'totem'

tests = {}
local tester = totem.Tester()
local precision = 1e-8

function tests.test_sampler()
    local dset = {data = torch.Tensor({{1}, {2}, {3}, {4}, {5}})}
    local td = dataset.TableDataset(dset)
    local sampler = td:sampler()
    local samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {1,2,3,4,5}, "sample a dataset")
end

function tests.test_binarize()
    local dset = {data = torch.Tensor({{1}, {2}, {3}, {4}, {5}})}
    local td = dataset.TableDataset(dset)
    local sampler = td:sampler()
    local samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {1,2,3,4,5}, "dataset wrong before binarization")

    td:binarize(3)
    sampler = td:sampler()
    samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {0,0,1,1,1}, "dataset wrong after binarization")
end

function tests.test_splitter()
    local data = torch.rand(5,10)
    local class = torch.Tensor({1,2,3,2,1})
    local td = dataset.TableDataset({data=data, class=class})
    local rval = dataset.splitter(td, {ratio=0.21})
    tester:assertTensorEq(rval[1].dataset.data, data[{{1,4}}], precision)
    tester:assertTensorEq(rval[2].dataset.data, data[{{4,5}}], precision)
    tester:assertTensorEq(rval[1].dataset.class, class[{{1,4}}], precision)
    tester:assertTensorEq(rval[2].dataset.class, class[{{4,5}}], precision)
end

function tests.test_splitter_noClass()
    local data = torch.rand(5,10)
    local td = dataset.TableDataset({data=data})
    local rval = dataset.splitter(td, {ratio=0.2})
    tester:assertTensorEq(rval[1].dataset.data, data[{{1,4}}], precision)
    tester:assertTensorEq(rval[2].dataset.data, data[{{4,5}}], precision)
end

return tester:add(tests):run()



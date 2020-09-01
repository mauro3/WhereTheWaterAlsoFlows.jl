# Download BedMachine
using Pkg
Pkg.activate(".")
using NCDatasets, Plots, Parameters


const rw = 1000.0
const ri = 910.0

!ispath("data") && mkdir("data")

url = "file://" * homedir() * "/itet-stor/glazioarch/GlacioData/BedMachine_Antarctica/168596330/BedMachineAntarctica_2019-11-05_v01.nc"

filename = if !isfile(joinpath("data", splitdir(url)[2]))
    download(url, joinpath("data", splitdir(url)[2]))
else
    joinpath("data", splitdir(url)[2])
end

ds = NCDataset(filename)

downscale = 2

x = ds["x"][1:downscale:end];
y = ds["y"][1:downscale:end];
x = x[1]:x[2]-x[1]:x[end]
y = y[1]:y[2]-y[1]:y[end]
mask = ds["mask"][1:downscale:end,1:downscale:end];
mask = (mask.==2) .| (mask.==4); # grounded ice or lake Vostok
surface = ds["surface"][1:downscale:end,1:downscale:end];
bed = ds["bed"][1:downscale:end,1:downscale:end];


surf = copy(surface);
surf[.!mask] .= NaN;

## Now run flow routing with surface and bed

using WhereTheWaterFlows, PyPlot
const WWF = WhereTheWaterFlows

@time area, slen, dir, nout, nin, pits, c, bnds = waterflows(rw*bed .+ (surf.-bed)*ri, drain_pits=false);

WWF.plotarea(1:size(area,1), 1:size(area,2), area, pits)

#

using WhereTheWaterAlsoFlows
#const WWAF = WhereTheWaterAlsoFlows
D = flow_routing2D(x, y, bed, surface.-bed);

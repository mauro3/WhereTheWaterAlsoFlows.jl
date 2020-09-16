# Download BedMachine
using Pkg
Pkg.activate(".")
using NCDatasets, Plots, Parameters

smooth_surface = 2000.0 # smoothing radius, in meters

const rw = 1000.0
const ri = 910.0

!ispath("../data") && mkdir("../data")
!ispath("../output") && mkdir("../output")

url = "file://" * homedir() * "/itet-stor/glazioarch/GlacioData/BedMachine_Antarctica/168596330/BedMachineAntarctica_2019-11-05_v01.nc"

filename = if !isfile(joinpath("../data", splitdir(url)[2]))
    print("Downloading data ...")
    download(url, joinpath("../data", splitdir(url)[2]))
    println("done.")
else
    joinpath("../data", splitdir(url)[2])
end

if !(@isdefined ds)
    print("Loading NC file ... ")
    ds = NCDataset(filename)

    # downscale = 2 finest possible on 16GB
    # downscale = 1 ok on 32GB ram (needs around 16GB)
    #
    downscale = 200

    x = ds["x"][1:downscale:end];
    y = ds["y"][1:downscale:end];
    x = x[1]:x[2]-x[1]:x[end]
    y = y[1]:y[2]-y[1]:y[end]
    mask_ = ds["mask"][1:downscale:end,1:downscale:end];
    mask = (mask_.==2) .| (mask_.==4) .| (mask_.==1); # route water over(under): grounded ice or lake Vostok or ice-free-land
    surface = ds["surface"][1:downscale:end,1:downscale:end];
    bed = convert(Matrix{Float32}, ds["bed"][1:downscale:end,1:downscale:end]);
    println("done")
end

if smooth_surface>0
    dn = round(Int, smooth_surface/step(x))
    if dn>0
        #smooth
        @warn "todo"
    end
end

surf = copy(surface);
surf[.!mask] .= NaN;

## Now run flow routing with surface and bed

using WhereTheWaterFlows, PyPlot
const WWF = WhereTheWaterFlows
# inds = [45:55,45:55]
# mask = mask[inds...]
# hydpot = (rw*bed .+ (surf.-bed)*ri)[inds...]

# hydpot = (rw*bed .+ (surf.-bed)*ri)
# @time area, slen, dir, nout, nin, pits, c, bnds = waterflows( hydpot, drain_pits=true, bnd_as_pits=true);
# area[.!mask] .= NaN;
# WWF.plotarea(1:size(area,1), 1:size(area,2), area, pits)
# if !all(getindex.(Ref(mask), pits).==0)
#     @warn "Not all pits were removed"
# end
#

using WhereTheWaterAlsoFlows
#const WWAF = WhereTheWaterAlsoFlows

D = flow_routing2D(x, reverse(y), bed, surface.-bed, mask, mask*100, plotyes=false);

nothing

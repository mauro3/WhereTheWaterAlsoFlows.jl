# Download BedMachine
using Pkg
Pkg.activate(".")
using NCDatasets, Plots, Parameters

!ispath("../data") && mkdir("../data")
!ispath("../output") && mkdir("../output")

# smooth_surface = 2000.0 # smoothing radius, in meters
nsm_s  = 2 # surface
nsm_b  = 0 # bed

run_id = "run_ds_20" # DEBUG: define the output directory within ../output/

!ispath("../output/$run_id") && mkdir("../output/$run_id")

const rw = 1000.0
const ri = 910.0

url = "file://" * homedir() * "/itet-stor/glazioarch/GlacioData/BedMachine_Antarctica/168596330/BedMachineAntarctica_2019-11-05_v01.nc"
# url = "file://" * "/Volumes/luraess/glazioarch/GlacioData/BedMachine_Antarctica/168596330/BedMachineAntarctica_2019-11-05_v01.nc"

filename = if !isfile(joinpath("../data", splitdir(url)[2]))
    print("Downloading data ...")
    download(url, joinpath("../data", splitdir(url)[2]))
    println("done.")
else
    joinpath("../data", splitdir(url)[2])
end

# if !(@isdefined ds) # DEBUG: should be done differently to allow for changing ds while the file already downloaded
    print("Loading NC file ... ")
    ds = NCDataset(filename)

    # downscale = 2 finest possible on 16GB
    # downscale = 1 ok on 32GB ram (needs around 16GB)
    downscale = 30

    x  = ds["x"][1:downscale:end]
    y  = ds["y"][1:downscale:end]
    x  = x[1]:x[2]-x[1]:x[end]
    y  = y[1]:y[2]-y[1]:y[end]
    mask_   = ds["mask"][1:downscale:end, 1:downscale:end]
    mask    = (mask_.==2) .| (mask_.==4) .| (mask_.==1) # route water over(under): grounded ice or lake Vostok or ice-free-land
    surface = ds["surface"][1:downscale:end, 1:downscale:end]
    bed     = convert(Matrix{Float32}, ds["bed"][1:downscale:end, 1:downscale:end])

    # Pad the data to make it GPU compatible with good performance # DEBUG: do this until ParallelStencil is fixed: see https://github.com/samo-lin/ParallelStencil.jl/issues/42
    olx = rem(length(x), 32)
    oly = rem(length(y), 8 )
    x1=1; xE=0; if (olx!=0) x1 = Int(ceil(olx/2)); xE = olx-x1+1; end
    y1=1; yE=0; if (oly!=0) y1 = Int(ceil(oly/2)); yE = oly-y1+1; end

    x  = x[x1:end-xE]
    y  = y[y1:end-yE]
    mask    = mask[x1:end-xE, y1:end-yE]
    surface = surface[x1:end-xE, y1:end-yE]
    bed     = bed[x1:end-xE, y1:end-yE]

    println("done")
# end
println("grid size: nx=$(length(x)), ny=$(length(y))")

# if smooth_surface>0
#     dn = round(Int, smooth_surface/step(x))
#     if dn>0
#         #smooth
#         @warn "todo"
#     end
# end

@views function smooth!(A)
    A[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0./6.1.*(diff(diff(A[:,2:end-1],dims=1),dims=1) .+ diff(diff(A[2:end-1,:],dims=2),dims=2))
    return
end

print("Smoothing DEM [surface $nsm_s step(s), bed $nsm_b step(s)] ... ")
for ism = 1:nsm_s
    smooth!(surface)
end
for ism = 1:nsm_b
    smooth!(bed)
end
println("done")

surf = copy(surface)
surf[.!mask] .= NaN

## Now run flow routing with surface and bed

# using WhereTheWaterFlows, PyPlot
# const WWF = WhereTheWaterFlows
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

@time D, F, xcp, ycp = flow_routing2D(x, reverse(y), bed, surface.-bed, mask, mask*1, outdir="../output/$run_id")

# visu
F[.!mask] .= NaN

plt = heatmap(xcp, ycp, log10.(F'), aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:viridis, framestyle=:box, title="log10 ||Flux||")

# savefig(plot(plt, dpi=300), joinpath(@__DIR__, "../output/$run_id/o$(length(x))_final.png"))

display(plt)

nothing

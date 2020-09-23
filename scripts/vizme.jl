const Dat = Float32
using Printf, Statistics, LinearAlgebra, DelimitedFiles, Plots
# pyplot()

run_id = "run_ds_40"

outdir = "../output/$run_id"
function load_array(A, A_name, extension, isave, outd)
    fid=open("$outd/$(isave)_$(A_name).$(extension)", "r"); read!(fid, A); close(fid)
end

Info = readdlm("$outdir/0_infos.inf")

Lx    = Info[1]
Ly    = Info[2]
nx    = Int(Info[3])
ny    = Int(Info[4])
isave = Int(Info[5])
err   = Info[6]

Mask = zeros(Dat, nx, ny)
xcp  = zeros(Dat, nx, 1 )
ycp  = zeros(Dat, ny, 1 )
F    = zeros(Dat, nx, ny)

load_array(Mask, "M"   , "res", 0, outdir)
load_array(xcp , "xcp" , "res", 0, outdir)
load_array(ycp , "ycp" , "res", 0, outdir)

load_array(F   , "F"   , "res", isave, outdir)

# visu
F[Mask.==0] .= NaN
xcp = xcp[:,1]
ycp = ycp[:,1]

plt = heatmap(xcp, ycp, log10.(F'), aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:viridis, framestyle=:box, title="log10 ||Flux|| (ϵ=$err)")

savefig(plot(plt, dpi=300), joinpath(@__DIR__, "$outdir/o$(length(xcp))_$isave.png"))

display(plt)

# Download BedMachine

using NCDatasets, Plots

!ispath("data") && mkdir("data")

url = "file://" * homedir() * "/itet-stor/glazioarch/GlacioData/BedMachine_Antarctica/168596330/BedMachineAntarctica_2019-11-05_v01.nc"

filename = if !isfile(joinpath("data", splitdir(url)[2]))
    download(url, joinpath("data", splitdir(url)[2]))
else
    joinpath("data", splitdir(url)[2])
end

ds = NCDataset(filename)

downscale = 100

x = ds["x"][1:downscale:end]
y = ds["y"][1:downscale:end]
mask = ds["mask"][1:downscale:end,1:downscale:end]
mask = (mask.==2) .| (mask.==4) # grounded ice or lake Vostok
surface = ds["surface"][1:downscale:end,1:downscale:end]
bed = ds["bed"][1:downscale:end,1:downscale:end]


surf = copy(surface)
surf[.!mask] .= NaN
heatmap(surf)

# now run flow routing with surface and bed

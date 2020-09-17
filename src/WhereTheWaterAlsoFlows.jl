__precompile__(false) # see https://github.com/samo-lin/ParallelStencil.jl/issues/39

module WhereTheWaterAlsoFlows

export flow_routing2D

const USE_GPU = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Plots, Printf, Statistics, LinearAlgebra

################################
@parallel function def_err!(errD::Data.Array, D::Data.Array)

    @all(errD) = @all(D)

    return
end

@parallel function chk_err!(errD::Data.Array, D::Data.Array)

    @all(errD) = @all(D) - @all(errD)

    return
end

@parallel function mask_D!(D::Data.Array, Mask::Data.Array)

    @all(D) = @all(D) * @all(Mask)

    return
end

@parallel function compute_ϕ!(ϕ::Data.Array, rdh::Data.Number, rr::Data.Number, Zb::Data.Array, D::Data.Array, H::Data.Array)

    @all(ϕ) = @all(Zb) + rdh*@all(D) + rr*@all(H)

    return
end

@parallel_indices (ix,iy) function compute_flux!(Ux::Data.Array, Uy::Data.Array, qDx::Data.Array, qDy::Data.Array, k::Data.Number, _dx::Data.Number, _dy::Data.Number, ϕ::Data.Array, D::Data.Array)

    if (ix<=size(Ux,1)  && iy<=size(Ux,2))    Ux[ix,iy] = -k*_dx*(ϕ[ix+1,iy]-ϕ[ix,iy]) end
    if (ix<=size(Uy,1)  && iy<=size(Uy,2))    Uy[ix,iy] = -k*_dy*(ϕ[ix,iy+1]-ϕ[ix,iy]) end
    if (ix<=size(qDx,1) && iy<=size(qDx,2))  qDx[ix,iy] = Ux[ix,iy]*(Ux[ix,iy]>0.0)*D[ix,iy] + Ux[ix,iy]*(Ux[ix,iy]<0.0)*D[ix+1,iy  ] end
    if (ix<=size(qDy,1) && iy<=size(qDy,2))  qDy[ix,iy] = Uy[ix,iy]*(Uy[ix,iy]>0.0)*D[ix,iy] + Uy[ix,iy]*(Uy[ix,iy]<0.0)*D[ix  ,iy+1] end

    return
end

@parallel function compute_∂D!(∂D::Data.Array, _dx::Data.Number, _dy::Data.Number, M::Data.Number, qDx::Data.Array, qDy::Data.Array)

    @inn(∂D) = -(_dx*@d_xi(qDx) + _dy*@d_yi(qDy)) + 1.0/M

    return
end

@parallel function update_D!(∂Ddτ::Data.Array, D::Data.Array, damp::Data.Number, dτ::Data.Number, ∂D::Data.Array)

    @all(∂Ddτ) =  damp*@all(∂Ddτ) + @all(∂D)
    @all(D)    =  @all(D) + dτ*@all(∂Ddτ)

    return
end

@parallel_indices (ix,iy) function set_BCx!(D::Data.Array, val_BC::Data.Number)

    if (ix==1         && iy<=size(D,2)) D[ix,iy] = val_BC end
    if (ix==size(D,1) && iy<=size(D,2)) D[ix,iy] = val_BC end

    return
end

@parallel_indices (ix,iy) function set_BCy!(D::Data.Array, val_BC::Data.Number)
    # fixed height
    if (ix<=size(D,1) && iy==1        ) D[ix,iy] = val_BC end
    if (ix<=size(D,1) && iy==size(D,2)) D[ix,iy] = val_BC end
    # no flux
    # if (ix<=size(D,1) && iy==1        ) D[ix,iy] = D[ix,iy+1] end
    # if (ix<=size(D,1) && iy==size(D,2)) D[ix,iy] = D[ix,iy-1] end
    return
end

# physics scales
const ρ̂i  = 910
const ρ̂w  = 1000
const ĝ   = 9.81
const ϕ̂   = 1e7
const x̂   = 1e6
const M̂   = 1e-6
const D̂   = 1
# physics dependent
const t̂   = D̂/M̂
const Ĥ   = ϕ̂/ρ̂w/ĝ
const Ψ̂   = ϕ̂/x̂
const û   = x̂/t̂
const k̂   = û/Ψ̂
const rr  = ρ̂i/ρ̂w
const rdh = D̂/Ĥ    # fixed ice thikness routing
const s2d = 3600*24 # seconds in a day

"""

Inputs:
- xc, yc: center coordinates (as a range)
- Zb, H: bed elevation, ice thickness
- D0: IC for water layer thickness (default ==1)
"""
@views function flow_routing2D(xc::AbstractRange, yc::AbstractRange, Zb, H, Mask=zero(Zb).+1, D0=zero(Zb).+1;
                               plotyes=true, outdir="")

    nx     = length(xc)
    ny     = length(yc)

    @assert (nx,ny) == size(Zb) == size(H) == size(D0) "Sizes don't match"
    # fastest if multiple of 16 (as no overlength here)
    if USE_GPU && (rem(nx,16)!=0 || rem(ny,16)!=0)
        @warn "Gridpoints not divisible by 16, this leads to slower speed."
    end

    Lx, Ly = xc[end]-xc[1], yc[end]-yc[1]

    # nondim
    Lx, Ly = Lx/x̂, Ly/x̂
    xc, yc = xc/x̂, yc/x̂
    Zb     = Data.Array(Zb/Ĥ)
    H      = Data.Array(H/Ĥ)
    D      = Data.Array(D0/D̂)
    Mask   = Data.Array(Mask)
    # parameters
    k      = 0.1
    M      = M̂*s2d
    # numerics
    nt     = 1
    itrMax = 1e5 #1e8
    nout   = 1000
    nmax   = 100
    ε      = 1e-8     # aboslute tolerance
    damp   = 0.94  #0.9 for nx=100
    # dτ_sc  = nx/128*30.0 #60 for nx,ny=256 #30 for nx,ny=128 #24 for nx,ny=96 #12.0 for nx,ny=64
    dτ_sc  = nx/128*5.0 #60 for nx,ny=256 #30 for nx,ny=128 #24 for nx,ny=96 #12.0 for nx,ny=64
    dx, dy = Lx/nx, Ly/ny
    _dx, _dy = 1.0/dx, 1.0/dy
    # D      = max(Zb)-Zb + 10;
    ∂D     = @zeros(nx  ,ny  )
    ϕ      = @zeros(nx  ,ny  )
    Ux     = @zeros(nx-1,ny  )
    Uy     = @zeros(nx  ,ny-1)
    qDx    = @zeros(nx-1,ny  )
    qDy    = @zeros(nx  ,ny-1)
    ∂Ddτ   = @zeros(nx  ,ny  )
    errD   = @zeros(nx  ,ny  )
    err1=[]; err2=[]
    # action
    iter = 1; err = 2*ε;  max_U = 1.0
    @parallel mask_D!(D, Mask)
    while err > ε && iter < itrMax
        @parallel def_err!(errD, D)
    	# flow routing physics
    	@parallel compute_ϕ!(ϕ, rdh, rr, Zb, D, H)
    	@parallel compute_flux!(Ux, Uy, qDx, qDy, k, _dx, _dy, ϕ, D)
    	@parallel compute_∂D!(∂D, _dx, _dy, M, qDx, qDy)
    	if mod(iter,nmax)==0 || iter==1
    		max_U = max(maximum(abs.(Ux)), maximum(abs.(Uy)))
    	end
    	dτ = min(dx,dy)/max_U/dτ_sc
    	@parallel update_D!(∂Ddτ, D, damp, dτ, ∂D)
    	@parallel set_BCx!(D, 0.0)
    	@parallel set_BCy!(D, 0.0)
        @parallel mask_D!(D, Mask)
    	# Check errs
    	if mod(iter,nout)==0
    		@parallel chk_err!(errD, D)
    		# ermb  = 1.0./M*dx*dy*(nx-2)*(ny-2) - sum(abs.(qDx[[1, end],2:end-1])) - sum(abs.(qDy[2:end-1,[1, end]]))
    		push!(err1, maximum(abs.(errD[:])));# push!(err2, abs(ermb))
    		err   = err1[end]
    		# @printf("iter=%d  errD=%1.3e, errMB=%1.3e \n", iter, err1[end], err2[end])
            @printf("iter=%d  errD=%1.3e \n", iter, err1[end])
    	end
        D = D.*Mask
    	iter+=1
    end
    if plotyes
        Zb[D.==0.0] .= NaN
        D[D.==0.0] .= NaN
        # ploting
        xcp = xc*x̂/1e3; ycp = yc*x̂/1e3
        p1 = heatmap(xcp, ycp, Zb'*Ĥ, aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:inferno, title="Zb")
        # p1 = heatmap(qDx', aspect_ratio=1)
        p2 = heatmap(xcp, ycp, D'*D̂, aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:inferno, title="D")
        # p2 = heatmap(xcp, ycp, D'*D̂ + Zb'*Ĥ, aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:inferno, title="Zb+D")
        # p2 = heatmap(xcp, ycp, errD', aspect_ratio=1, xlims=(xcp[1], xcp[end]), ylims=(ycp[1], ycp[end]), c=:inferno, title="Zb+D")
        # p3 = plot(xcp[2:end-1], D[2:end-1,Int(round(ny/2))]*D̂, ylabel="D [m]", yscale=:log10, linewidth=2, framestyle=:box, legend=false)
        # l  = @layout [a b; c]
        # display(plot(p1, p2, p3, layout = l))
        display(plot(p1, p2))
        # savefig(plot(p1, p2, p3, layout = l), joinpath(@__DIR__, "../output/o$nx.png"))
    end
    return D * D̂, qDx, qDy, xcp, ycp
end

end

using Pkg; Pkg.activate("../..")

const USE_GPU  = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID   = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Test, Plots, Printf, Statistics, LinearAlgebra

################################
@parallel function swap_old!(D_o::Data.Array, ∂D_o::Data.Array, D::Data.Array, ∂D::Data.Array)

        @all(D_o)  = @all(D)
        @all(∂D_o) = @all(∂D)
    
    return
end

@parallel function def_err!(errD::Data.Array, D::Data.Array)

        @all(errD) = @all(D)
    
    return
end

@parallel function chk_err!(errD::Data.Array, D::Data.Array)

        @all(errD) = @all(D) - @all(errD)
    
    return
end


@parallel function compute_ϕ!(ϕ::Data.Array, rdh::Data.Number, rr::Data.Number, Zb::Data.Array, D::Data.Array, H::Data.Array)

    @all(ϕ) = @all(Zb) + rdh*@all(D) + rr*@all(H)
    
    return
end

@parallel function compute_ϕ!(ϕ::Data.Array, rdh::Data.Number, rr::Data.Number, Zb::Data.Array, D::Data.Array, H::Data.Array)

    @all(ϕ) = @all(Zb) + rdh*@all(D) + rr*@all(H)
    
    return
end

@parallel_indices (ix,iy) function compute_flux!(Ux::Data.Array, Uy::Data.Array, qDx::Data.Array, qDy::Data.Array, k::Data.Number, _dx::Data.Number, _dy::Data.Number, ϕ::Data.Array, D::Data.Array, Ux::Data.Array, Uy::Data.Array)

    if (ix< nx && iy<=ny)  Ux[ix,iy]  = -k*_dx*(ϕ[ix+1,iy]-ϕ[ix,iy]) end
    if (ix<=nx && iy< ny)  Uy[ix,iy]  = -k*_dy*(ϕ[ix,iy+1]-ϕ[ix,iy]) end
    if (ix< nx && iy<=ny)  qDx[ix,iy] = Ux[ix,iy]*(Ux[ix,iy]>0.0)*D[ix,iy] + Ux[ix,iy]*(Ux[ix,iy]<0.0)*D[ix+1,iy  ] end
    if (ix<=nx && iy< ny)  qDy[ix,iy] = Uy[ix,iy]*(Uy[ix,iy]>0.0)*D[ix,iy] + Uy[ix,iy]*(Uy[ix,iy]<0.0)*D[ix  ,iy+1] end

    return
end

@parallel function compute_∂D!(∂D::Data.Array, _dx::Data.Number, _dy::Data.Number, M::Data.Number, qDx::Data.Array, qDy::Data.Array)

    @inn(∂D) = -(_dx*@d_xi(qDx) + _dy*@d_yi(qDy)) + 1.0/M
    
    return
end

@parallel function update_D!(∂Ddτ::Data.Array, D::Data.Array, damp::Data.Number, dt::Data.Number, cn::Data.Number, dτ::Data.Number, D_o::Data.Array, ∂D::Data.Array, ∂D_o::Data.Array)

    @all(∂Ddτ) =  damp*@all(∂Ddτ) + ( -(@all(D)-@all(D_o))/dt + ((1-cn)*@all(∂D) + cn*@all(∂D_o)) )
    @all(D)    =  @all(D) + dτ*@all(∂Ddτ)
    
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
const s2d = 3600*24

@views function flow_routing2D()
# nondim
Lx, Ly = 10e3/x̂, 10e3/x̂
k      = 0.1
ttot   = 1000e10*s2d/t̂ # ttot>>1 && nt=1 steady state
M      = M̂*s2d
tim_p  = 0.0
# numerics
nx     = 96 # fastest if multiple of 16 (as no overlength here)
ny     = 96 # fastest if multiple of 16 (as no overlength here)
nt     = 1
nout   = 1000
ε      = 1e-10     # aboslute tolerance
damp   = 0.9
dτ_sc  = 12.0
cn     = 0.0       # crank-Nicolson if cn=0.5 (transient)
dx, dy = Lx/nx, Ly/ny
_dx, _dy = 1.0/dx, 1.0/dy
xc     = dx/2:dx:Lx-dx/2
yc     = dy/2:dy:Ly-dy/2
dt     = ttot/nt
# initial
Zb     = (Lx.-xc)./Lx./100.0 .- 30.0./Ĥ.*exp.(-(xc.-Lx/2.0).^2/(Lx/8.0)^2 -(yc'.-Ly/2.0).^2/(Ly/8.0)^2)
H      = 100.0./Ĥ .- xc.*3.0 + 0*yc'
Zb     = Data.Array(Zb)
H      = Data.Array(H)
# H      = (Lx-xc)/Lx/100 + 30/H_s*exp(-(xc-Lx/2).^2/(Lx/8)^2) + (xc + 50)/H_s;
D      = 1.0.*@ones(nx,ny)
# D      = max(Zb)-Zb + 10;
∂D     = @zeros(nx  ,ny  )
D_o    = @zeros(nx  ,ny  )
∂D_o   = @zeros(nx  ,ny  )
ϕ      = @zeros(nx  ,ny  )
Ux     = @zeros(nx-1,ny  )
Uy     = @zeros(nx  ,ny-1)
qDx    = @zeros(nx-1,ny  )
qD     = @zeros(nx  ,ny-1)
RD     = @zeros(nx  ,ny  )
∂Ddτ   = @zeros(nx  ,ny  )
errD   = @zeros(nx  ,ny  )
err1=[]; err2=[]
# action
for it = 1:nt
    @parallel swap_old!(D_o, ∂D_o, D, ∂D)

    iter = 1; err = 2*ε
    while err > ε
        @parallel def_err!(errD, D)
        # flow routing physics
        @parallel compute_ϕ!(ϕ, rdh, rr, Zb, D, H)

        @parallel compute_flux!(Ux, Uy, qDx, qDy, k, _dx, _dy, ϕ, D, Ux, Uy)

        max_U = max(maximum(abs.(Array(Ux))), maximum(abs.(Array(Uy))))
        dτ    = 1.0/(1.0/dt + 1.0/(min(dx,dy)/max_U/dτ_sc))

        @parallel update_D!(∂Ddτ, D, damp, dt, cn, dτ, D_o, ∂D, ∂D_o)

        D[[1,end]]  .= 0.0 # BC

        # Check errs
        if mod(iter,nout)==0
            @parallel chk_err!(errD, D)
            ermb  = 1.0./M*dx*dy*(nx-2)*(ny-2) - sum(abs.(Array(qDx[[1, end],:]))) - sum(abs.(Array(qDy[:,[1, end]])))
            push!(err1, maximum(abs.(Array(errD)))); push!(err2, abs(ermb))
            err   = err1[end]
            @printf("iter=%d  errD=%1.3e, errMB=%1.3e \n", iter, err1[end], err2[end])
        end
        iter+=1
    end
    tim_p = tim_p + dt
    # ploting
    default(size=(700,800))
    p1 = plot(xc*x̂/1e3, Zb*Ĥ, label="Zb", linewidth=2)
        plot!(xc*x̂/1e3, D*D̂+(H+Zb)*Ĥ, label="D+Zb+H", linewidth=2, title=string("Time = ",tim_p*t̂/s2d," days"), framestyle=:box)
    p2 = plot(xc*x̂/1e3, ϕ, ylabel="ϕ", linewidth=2, framestyle=:box, legend=false)
    p3 = plot(xc[2:end-1]*x̂/1e3, D[2:end-1]*D̂, ylabel="D [m]", yscale=:log10, linewidth=2, framestyle=:box, legend=false)
    p4 = plot(xc*x̂/1e3, D*D̂ + Zb*Ĥ, label="D+Zb [m]", linewidth=2)
        plot!(xc*x̂/1e3, Zb*Ĥ, label="Zb [m]", linewidth=2, framestyle=:box)
    p5 = plot(xc[2:end]*x̂/1e3, U*û, ylabel="U", linewidth=2, framestyle=:box, legend=false)
    p6 = plot(xc[2:end]*x̂/1e3, qD, xlabel="x [km]", ylabel="q", linewidth=2, framestyle=:box, legend=false)
    display(plot(p1, p2, p3, p4, p5, p6, layout=(6,1)))
    # figure(2),clf
    # semilogy(err1,'Linewidth',LW),hold on
    # semilogy(err2,'Linewidth',LW),hold off,legend('errD','errMB')
end

return
end

@time flow_routing2D()
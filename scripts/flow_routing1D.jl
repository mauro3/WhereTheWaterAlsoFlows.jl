# using Pkg; Pkg.activate("../..")

# Implement routing with a FVM
using Test, Plots, Printf, Statistics

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

@views function flow_routing1D()
# nondim
Lx     = 10e3/x̂
k      = 0.1
ttot   = 1000e10*s2d/t̂ # ttot>>1 && nt=1 steady state
M      = M̂*s2d
tim_p  = 0.0
# numerics
nx     = 100
nt     = 1
nout   = 1000
ε      = 1e-10     # aboslute tolerance
damp   = 0.9
dτ_sc  = 12.0
cn     = 0.0       # crank-Nicolson if cn=0.5 (transient)
dx     = Lx/nx
xc     = dx/2:dx:Lx-dx/2
dt     = ttot/nt
# initial
Zb     = (Lx.-xc)./Lx./100.0 .- 30.0./Ĥ.*exp.(-(xc.-Lx/2.0).^2/(Lx/8.0)^2)
H      = 100.0./Ĥ .- xc.*3.0
# H      = (Lx-xc)/Lx/100 + 30/H_s*exp(-(xc-Lx/2).^2/(Lx/8)^2) + (xc + 50)/H_s;
D      = 1.0.*ones(nx)
# D      = max(Zb)-Zb + 10;
∂D     = zeros(nx)
D_o    = zeros(nx)
∂D_o   = zeros(nx)
ϕ      = zeros(nx)
U      = zeros(nx-1)
qD     = zeros(nx-1)
RD     = zeros(nx)
∂Ddτ   = zeros(nx)
errD   = zeros(nx)
err1=[]; err2=[]
# action
for it = 1:nt
    D_o  .= D
    ∂D_o .= ∂D
    iter = 1; err = 2*ε
    while err > ε
        errD .= D
        # flow routing physics
        # if rout==1
        ϕ           .= Zb .+ rdh.*D .+ rr.*H              # fixed ice thikness routing
        # elseif rout==2
        # Zs          = Zb + H;                           # fixed ice surface routing
        # phi         = Zs + (1 - rr)*Zb + rdz*(1-rr)*D   # fixed ice surface routing
        # end
        U           .= -k*diff(ϕ)./dx
        qD          .= U.*(U.>0.0).*D[1:end-1] .+ U.*(U.<0.0).*D[2:end] # not working without upwind: qD = U.*(D(1:end-1)-D(2:end)).*0.5;
        ∂D[2:end-1] .= .-diff(qD)./dx .+ 1.0/M
        RD          .= .-(D.-D_o)./dt .+ ((1-cn).*∂D + cn*∂D_o)         # transient included if dt<inf
        # solver
        ∂Ddτ        .= damp*∂Ddτ .+ RD                                  # damping the rate of change
        dτ           = 1.0/(1.0/dt + 1.0/(dx/maximum(U)/dτ_sc))         # iterative timestep
        D           .= D .+ dτ.*∂Ddτ                                    # update D
        D[[1,end]]  .= 0.0 # BC
        # Check errs
        if mod(iter,nout)==0
            errD .= D.-errD
            ermb  = 1.0./M*dx*(nx-2) - sum(abs.(qD[[1, end]]))
            push!(err1, maximum(abs.(errD))); push!(err2, abs(ermb))
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

@time flow_routing1D()

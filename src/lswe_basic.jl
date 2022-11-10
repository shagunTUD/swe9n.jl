using Gridap
using Printf

include("dispersion.jl")

#cd( pwd() * "/examples/swe9n.jl" )


# Constants
g = 9.81;


# Generate Cartesian Domain 2DH 
domL = 300.0
domW = 10.0
dx = 1.25
domain = (0,domL,0,domW)
partition = (Int(domL/dx), Int(domW/dx))
model = CartesianDiscreteModel(domain, partition)
writevtk(model, "test/model")


# Label sides of the domain
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "inlet", ["tag_7","tag_1","tag_3"])
add_tag_from_tags!(labels, "outlet", ["tag_8","tag_2","tag_4"])
add_tag_from_tags!(labels, "sideWall", ["tag_5","tag_6"])
writevtk(model, "test/model")


# Define wave conditions
T = 10.0; #s
h = 1.0; #m
H = 0.02; #m
L = dispersionRel(g, h, T)
k = 2*π/L;
ω = 2*π/T;
println("Celerity ",ω/k)
println("Courant ",ω/k/dx*0.01)


# Define Test Fnc
# ---------------------Start---------------------
reffeη = ReferenceFE(lagrangian, Float64, 1)
Ψ1 = TestFESpace(model, reffeη, 
    conformity=:H1, dirichlet_tags=["inlet"])

reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
Ψ2 = TestFESpace(model, reffeu, 
    conformity=:H1, 
    dirichlet_tags=["inlet", "outlet","sideWall"],
    dirichlet_masks=[(true,false), (true,false), (false,true)])
# ----------------------End----------------------


# Define Trial Space
# ---------------------Start---------------------
g1(x, t::Real) = H/2.0*sin(-ω*t)
g1(t::Real) = x -> g1(x,t)
Eta = TransientTrialFESpace(Ψ1, g1)

g2a(x, t::Real) = VectorValue(H/2.0*sin(-ω*t)*ω/k, 0.0)
g2a(t::Real) = x -> g2a(x,t)
g2b(x, t::Real) = VectorValue(0.0, 0.0)
g2b(t::Real) = x -> g2b(x,t)
g2c(x, t::Real) = VectorValue(0.0, 0.0)
g2c(t::Real) = x -> g2c(x,t)
U = TransientTrialFESpace(Ψ2, [g2a,g2b,g2c])

Y = MultiFieldFESpace([Ψ1, Ψ2])
X = TransientMultiFieldFESpace([Eta, U])
# ----------------------End----------------------


# Define integration space
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)


# Weak form
# ---------------------Start---------------------
res(t, (η, u), (ψ1, ψ2)) =
    ∫( ∂t(η)*ψ1 - (∇(ψ1)⋅u)*h )*dΩ + 
    ∫( ∂t(u)⋅ψ2 + (∇(η)⋅ψ2)*g )*dΩ

op = TransientFEOperator(res, X, Y)
# ----------------------End----------------------

##
# Solver
lin_solver = LUSolver()
Δt = 0.2
ode_solver = ThetaMethod(lin_solver, Δt, 0.5)


# Initial soln
t0 = 0.0
# The following form throws errors
η0 = interpolate_everywhere(0.0, Eta(t0))
u0 = interpolate_everywhere(VectorValue(0.0,0.0), U(t0))
#ηht = Gridap.solve(ode_solver, op, (η0,u0), t0, 100)
# Use the following forms instead
#x0 = interpolate_everywhere([0.0, VectorValue(0.0,0.0)], X(t0))
solnht = Gridap.solve(ode_solver, op, (η0,u0), t0, 100)
#solnht = Gridap.solve(ode_solver, op, x0, t0, 100)
#writevtk(Ω, "test/output", cellfields=["eta"=>η0, "u"=>u0])


##
# Execute
createpvd("test/sweTrans") do pvd    
    cnt=0
    for (solh, t) in solnht                       
        cnt = cnt+1
        ηh, uh = solh
        tval = @sprintf("%5.3f",t)
        if(cnt%10 != 0) 
            continue
        end
        pvd[t] = createvtk(Ω,"test/sweTrans_$tval"*".vtu",
            cellfields=["eta"=>ηh, "u"=>uh])            
    end
end


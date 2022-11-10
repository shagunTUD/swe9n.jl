using Gridap
using Printf
using LineSearches: BackTracking
using CSV
using Tables

include("dispersion.jl")

function lswe_setup(domX, domY, dx, dy, inletEta, inletU,
    simT, simΔt, outΔt, probesxy, probname)

    # Generate Cartesian Domain 2DH 
    domain = (domX[1], domX[2], domY[1], domY[2])
    partition = ( Int((domX[2]-domX[1])/dx), 
        Int((domY[2]-domY[1])/dy))
    model = CartesianDiscreteModel(domain, partition)


    # Label sides of the domain
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7,1,3])
    add_tag_from_tags!(labels, "outlet", ["tag_8","tag_2","tag_4"])
    add_tag_from_tags!(labels, "sideWall", ["tag_5","tag_6"])
    writevtk(model, "test/model")


    # Define Test Fnc
    # ---------------------Start---------------------
    reffeη = ReferenceFE(lagrangian, Float64, 1)
    Ψ1 = TestFESpace(model, reffeη, 
        conformity=:H1, dirichlet_tags=["inlet"])

    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Ψ2 = TestFESpace(model, reffeu, 
        conformity=:H1, 
        dirichlet_tags=["inlet", "outlet","sideWall"],
        dirichlet_masks=[(true,true), (true,false), (false,true)])
    # ----------------------End----------------------


    # Define Trial Space
    # ---------------------Start---------------------    
    Eta = TransientTrialFESpace(Ψ1, inletEta)
    
    g2b(x, t::Real) = VectorValue(0.0, 0.0)
    g2b(t::Real) = x -> g2b(x,t)
    g2c(x, t::Real) = VectorValue(0.0, 0.0)
    g2c(t::Real) = x -> g2c(x,t)
    U = TransientTrialFESpace(Ψ2, [inletU,g2b,g2c])

    Y = MultiFieldFESpace([Ψ1, Ψ2])
    X = TransientMultiFieldFESpace([Eta, U])    
    # ----------------------End----------------------


    # Water depth
    ha(x) = h 
    h0 = interpolate_everywhere(ha, FESpace(model,reffeη))
    # h0 = interpolate_everywhere(h, FESpace(model,reffeη))
    # Alternatively can do the above line
    # println(h0(VectorValue(1.0,2.0)))

    # Define integration space
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)


    # Weak form
    # ---------------------Start---------------------
    res(t, (η, u), (ψ1, ψ2)) =
        ∫( ∂t(η)*ψ1 - (∇(ψ1)⋅(u*h0)) )*dΩ + 
        ∫( ∂t(u)⋅ψ2 + (∇(η)⋅ψ2)*g )*dΩ

    op = TransientFEOperator(res, X, Y)
    # ----------------------End----------------------
    
    
    # Solver
    lin_solver = LUSolver()
    ode_solver = ThetaMethod(lin_solver, simΔt, 0.5)


    # Initial soln
    t0 = 0.0
    x0 = interpolate_everywhere([0.0, VectorValue(0.0,0.0)], X(t0))
    solnht = solve(ode_solver, op, x0, t0, simT)    

    numP = length(probesxy)    
    probeDa = zeros(Float64, 1, numP*5+1)    
    lDa = zeros(Float64, 1, numP*5+1)
    probeDa[1,1] = t0

    # Execute
    createpvd(probname) do pvd    
        cnt=0
        for (solh, t) in solnht                       
            cnt = cnt+1
            ηh, uh = solh
            tval = @sprintf("%5.3f",t)                
            println("Time : $tval")

            lDa[1] = t
            for ip in 1:numP
                ipxy = probesxy[ip];
                ik = (ip-1)*5 + 1
                lDa[ik+1] = ipxy[1]
                lDa[ik+2] = ipxy[2]
                lDa[ik+3] = ηh(ipxy)
                ipuh = uh(ipxy)
                lDa[ik+4] = ipuh[1]
                lDa[ik+5] = ipuh[2]
            end
            probeDa = vcat(probeDa, lDa);

            if(cnt%10 != 0) 
                continue
            end
            pvd[t] = createvtk(Ω,probname*"_$tval"*".vtu",
                cellfields=["eta"=>ηh, "u"=>uh, "h0"=>h0])            
        end
    end 
    
    CSV.write(probname*"_probes.csv", Tables.table(probeDa))
end


##
# Run simulation
g = 9.81
T = 10.0; #s
h = 1.0; #m
H = 0.2; #m
L = dispersionRel(g, h, T)
k = VectorValue(2*π/L*cos(0), 2*π/L*sin(0))
ω = 2*π/T;
probname = joinpath("test/lswe_W04")

probesxy = [Point(12.0, 5.0)
            Point(24.0, 5.0)
            Point(36.0, 5.0)
            Point(48.0, 5.0)];

g1(x, t::Real) = H/2.0*sin(k⋅x - ω*t)
g1(t::Real) = x -> g1(x,t)
g2a(x, t::Real) = H/2.0*sin(k⋅x - ω*t)*ω/norm(k)^2*k
g2a(t::Real) = x -> g2a(x,t)
lswe_setup((0,300), (0,10), 1.25, 1.25, g1, g2a,
    80, 0.2, 2, probesxy, probname)
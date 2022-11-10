using Roots: find_zero

function dispersionRel(g, h, T)

    f(L) = L-(g/2/π*T*T*tanh(2*π/L*h))
    L = find_zero(f,0)

    println("Time Period \t T \t ",T)
    println("Water Depth \t h \t ",h)
    println("Wave Length \t L \t ",L)
    println("Wave Celerity \t C \t ",L/T)
    println("Disp. Regime \t h/L \t ",round.(h/L; digits=3))
    
    return L
end

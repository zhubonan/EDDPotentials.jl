#=
Optimisation algorithms
=#
using LinearAlgebra

include("tpsd.jl")

"""

    opt_tpsd(vc::AbstractCalc; f_tol=1e-3, s_tol=1e-3, e_tol=1e-5, 
                  itermax=1000, trace=false, α_tol=1e-10, callback=nothing, 
                  trajectory=nothing)

Optimise a structure (calculation) using TPSD.
Reference:
    - Barzilai, J. and Borwein, J.M., 1988. Two-point step size gradient methods. IMA journal of numerical analysis, 8(1), pp.141-148

"""
function opt_tpsd(vc::AbstractCalc; f_tol=1e-4, s_tol=1e-6, e_tol=1e-5, 
                  itermax=1000, trace=false, α_tol=1e-10, callback=nothing, 
                  trajectory=nothing)
    f = get_forces(vc)
    f0 = similar(f)
    e0 = -9999.
    x = get_positions(vc)
    step = similar(x)
    x0 = x

    # Initial step
    α = 1e-8
    dx = similar(x)
    df = similar(f) 

    converged = false
    i = 1

    if !isa(vc, VariableCellCalc)
        s_tol = floatmax()
    end

    while i < itermax

        e = get_energy(vc)
        f = get_forces(vc)

        if isa(vc, VariableCellCalc)
            fa = get_forces(vc.calc)
            sa = get_stress(vc)
        else
            fa = f
            sa = get_stress(vc)
        end

        fmax = maximum(norm.(eachcol(fa)))
        smax = maximum(abs.(sa))
        de = abs(e - e0)

        # Check for convergence criteria
        if fmax < f_tol && smax < s_tol && de < e_tol 
            converged = true
        end

        #  Compute the step size
        dx .= x .- x0
        df .= f .- f0
        if i > 1
            α =  abs(dot(dx, df) / (dot(df, df) + floatmin(eltype(x))))
        end

        # Show trace information
        if trace
            @info @sprintf("Iteration %d  |F|= %-10.5g |Smax|= %-10.5g dE= %-10.5g α= %-10.5g |step|= %-10.5g",
                           i, fmax, smax, de, α, norm(step))
        end

        # Check if Alpha is too small
        if i > 10 && α < α_tol
            converged = false
            break
        end
    
        # Check if we are converged
        converged && break

        # Update variables 
        copy!(x0, x)
        copy!(f0, f)
        e0 = e

        # Compute step size 
        step .= f .* α

        # Apply the step
        x = x .+ step
        set_positions!(vc, x)

        # callback function
        isnothing(callback) || callback()

        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))

        i += 1
    end
    converged
end
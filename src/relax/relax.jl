using LinearAlgebra
#=
Relaxation of a structure
=#

using Printf

@with_kw struct Relax{T,M}
    calc::T
    method::M
    energy_threshold::Float64 = 1e-5
    force_threshold::Float64 = 1e-2
    stress_threshold::Float64 = 1e-1
    trajectory::Vector{Any} = []
    external_pressure::Matrix{Float64}
    relax_cell::Bool = true
    keep_trajectory::Bool = false
    verbose::Bool = true
    iterations::Int = 1000
end

function Relax(calc, method=TwoPointSteepestDescent(); external_pressure=0., kwargs...)
    if isa(external_pressure, Matrix)
        external_pressure_matrix = external_pressure
    else
        external_pressure_matrix = diagm([external_pressure, external_pressure, external_pressure])
    end
    Relax(;calc, method, external_pressure=external_pressure_matrix, kwargs...)
end

function relax!(re::Relax)
    (;calc, method, energy_threshold, force_threshold, stress_threshold) = re
    (;external_pressure, relax_cell, trajectory, keep_trajectory, verbose) = re
    (;iterations) = re

    # Initial position vector
    
    if relax_cell
        _calc = VariableCellCalc(calc;external_pressure)
    else
        _calc = calc
    end
    p0 = get_positions(_calc)[:]

    "Energy"
    function fo(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        get_energy(calc)
    end

    "Gradient"
    function go(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        if keep_trajectory
            cell = deepcopy(get_cell(calc))
            cell.metadata[:enthalpy] = get_energy(calc)
            cell.arrays[:forces] = get_forces(calc)
            push!(trajectory, cell)
        end
        forces = get_forces(calc)
        # ∇E = -F
        forces .* -1
    end

    start_time = time()
    "Callback function to control the stopping of the optimisation"
    last_energy = get_energy(_calc)

    """
        callback(x)

    Callback function to print relaxation information as well as checking the convergence
    as specified.
    """
    function callback(x)
        eng = get_energy(_calc)
        de = abs((eng - last_energy) / length(get_cell(_calc)))
        last_energy = eng
        forces = get_forces(_calc)
        fmax = maximum(norm.(eachcol(forces)))
        stress = eVAngToGPa.(get_stress(calc))
        smax = maximum(abs.(stress))
        elapsed = time() - start_time

        # Check for convergence criteria
        fmax < force_threshold ? fok = "T" : fok = "F"
        de < energy_threshold ? eok = "T" : eok = "F"
        smax < stress_threshold ? sok = "T" : sok = "F"

        # Print basic information
        if verbose
            if x.iteration == 0
                @printf "* Iter                    E               dE         |F|        Smax       Time         Conv\n"
                #        * Iter 0000: -7.85900821e+02  6.82121026e-14 6.63946e-03 4.20335e-02      0.222   |F|:T dE:T Smax:T
            end
            @printf "* Iter %04d: %15.8e %15.8e %10.5e %10.5e %10.3f   " x.iteration eng de fmax smax elapsed
            @printf "dE:%s |F|:%s Smax:%s\n" eok fok sok
        end


        # Break if the convergence criteria is reached
        if relax_cell
            if (fok == "T") && (eok == "T") && (sok == "T")
                return true
            end
        else
            if (fok == "T") && (eok == "T") 
                return true
            end
        end
        return false
    end

    res = optimize(
        x -> fo(x, _calc),
        x -> go(x, _calc),
        p0,
        method,
        Optim.Options(;
        callback,
        iterations,
        );
        inplace=false,
    )
end


"""
    relax!(calc::NNCalc;relax_cell=true, show_trace, method, opt_kwargs...)

Relax the structure of the calculator.
"""
function relax!(
    calc::NNCalc;
    relax_cell=true,
    pressure_gpa=0.0,
    show_trace=false,
    method=TwoPointSteepestDescent(),
    out_label="eddp-output",
    opt_kwargs...,
)

    if relax_cell
        p =  pressure_gpa / 160.21766208
        ext = diagm([p, p, p])
        vc = EDDP.VariableCellCalc(calc, external_pressure=ext)
        # Run optimisation
        res = EDDP.optimise!(vc; show_trace, method, opt_kwargs...)
    else
        vc = calc
        res = EDDP.optimise!(calc; show_trace, method, opt_kwargs...)
    end
    update_metadata!(vc, out_label)
    vc, res
end


"""
    optimise!(calc::AbstractCalc)

Optimise the cell using the Optim interface. Collect the trajectory if requested.
Note that the trajectory is collected for all force evaluations and may not 
corresponds to the actual iterations of the underlying LBFGS iterations.
"""
function optimise!(
    calc::AbstractCalc;
    show_trace=false,
    g_abstol=1e-6,
    f_reltol=0.0,
    successive_f_tol=2,
    traj=nothing,
    method=TwoPointSteepestDescent(),
    kwargs...,
)
    p0 = get_positions(calc)[:]

    "Energy"
    function fo(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        get_energy(calc)
    end

    "Gradient"
    function go(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        if !isnothing(traj)
            cell = deepcopy(get_cell(calc))
            cell.metadata[:enthalpy] = get_energy(calc)
            cell.arrays[:forces] = get_forces(calc)
            push!(traj, cell)
        end
        forces = get_forces(calc)
        # ∇E = -F
        # Collect the trajectory if requested
        forces .* -1
    end
    res = optimize(
        x -> fo(x, calc),
        x -> go(x, calc),
        p0,
        method,
        Optim.Options(;
            show_trace=show_trace,
            g_abstol,
            f_reltol,
            successive_f_tol,
            kwargs...,
        );
        inplace=false,
    )
    res
end


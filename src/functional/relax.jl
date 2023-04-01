using LinearAlgebra
#=
Relaxation of a structure
=#

using Printf
using Optim
using Configurations

@option struct RelaxOptions
    method::String="tpsd"
    energy_threshold::Float64 = 1e-5
    force_threshold::Float64 = 1e-2
    stress_threshold::Float64 = 1e-1
    trajectory::Vector{Any} = []
    external_pressure::Vector{Float64} = [0.]
    relax_cell::Bool = true
    keep_trajectory::Bool = false
    verbose::Bool = true
    iterations::Int = 1000
end

@with_kw struct Relax{T}
    calc::T
    options::RelaxOptions
end

struct RelaxResult
    calc::AbstractCalc
    dE::Float64
    fmax::Float64
    smax::Float64
    iterations::Int
    converged::Bool
end

function Base.show(io::IO, ::MIME"text/plain", x::RelaxResult)
    println(io, "RelaxResult:")
    println(io, "   $(get_cell(x.calc))")
    println(io, "   dE         = $(x.dE)")
    println(io, "   fmax       = $(x.fmax)")
    println(io, "   smax       = $(x.smax)")
    println(io, "   iterations = $(x.iterations)")
    println(io, "   converged  = $(x.converged)")
end

"""
    stress_matrix(vector=AbstractVector)

Build a external stress matrix from vector inputs.
"""
function external_stress_matrix(vector::AbstractVector)
    @assert length(vector) in [6, 1] "Expect a vector with length 6 or 1"
    if length(vector) == 1
        a = vector[1]
        return diagm([a,a,a])
    elseif  length(vector) == 6
        a, b, c, d, e, f = vector
        return [
            a b c
            b d e
            c e f
        ]
    end
end
external_pressure_matrix(x::Real) = external_pressure_matrix([x])

Relax(calc, options=RelaxOptions()) = Relax(calc, options)

"""
    multirelax!(re::Relax;itermax=2000, restartmax=5)

Perform relaxation and restart if it does not converge. This is useful when the
algorithm suffer from vanishing step sizes and a reset of the state of the optimiser
helps.
"""
function multirelax!(re::Relax;max_iter=2000, max_restart=5)
    local outcome
    itertotal = 0
    for _ in 1:max_restart 
        outcome = relax!(re)
        itertotal += outcome.iterations
        if outcome.converged || itertotal >= max_iter
            break
        end
    end
    RelaxResult(
        outcome.calc,
        outcome.dE,
        outcome.fmax,
        outcome.smax,
        itertotal,
        outcome.converged
    )
end


"""
    relax!(re::Relax)

Perform relaxation for a `Relax` object.
"""
function relax!(re::Relax)
    (;method, energy_threshold, force_threshold, stress_threshold) = re.options
    (;external_pressure, relax_cell, trajectory, keep_trajectory, verbose) = re.options
    (;iterations) = re.options
    calc = re.calc

    if method == "tpsd"
        _method = TwoPointSteepestDescent()
    elseif method == "lbfgs"
        _method = LBFGS(;
        linesearch=Optim.LineSearches.BackTracking(maxstep=0.2)
        )
    elseif method == "bfgs"
        _method = BFGS(;
        linesearch=Optim.LineSearches.BackTracking(maxstep=0.2)
        )
    elseif method == "cg"
        _method = ConjugateGradient(;
        linesearch=Optim.LineSearches.BackTracking(maxstep=0.2)
        )
    else
        throw(KeyError("Unknown `method`: $(method)"))
    end

    smat = external_stress_matrix(external_pressure)

    # Initial position vector
    
    if relax_cell
        _calc = VariableCellCalc(calc;external_pressure=smat)
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

    smax = 0.
    fmax = 0.
    de = 0.
    converged = false

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
                converged = true
                return true
            end
        else
            if (fok == "T") && (eok == "T") 
                converged = true
                return true
            end
        end
        return false
    end

    res = optimize(
        x -> fo(x, _calc),
        x -> go(x, _calc),
        p0,
        _method,
        Optim.Options(;
        callback,
        iterations,
        );
        inplace=false,
    )

    if !relax_cell
        smax = -1.
    end


    RelaxResult(
        calc,
        de,
        fmax,
        smax,
        res.iterations,
        converged,
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



# """
#     relax_structures(files, outdir, cf::CellFeature, ensemble::AbstractNNInterface;

# Relax many structures and place the relaxed structures in the output directory
# """
# function relax_structures(
#     files,
#     outdir,
#     cf::CellFeature,
#     ensemble::AbstractNNInterface;
#     nmax=500,
#     core_size=1.0,
#     relax_cell=true,
#     pressure_gpa=0.0,
#     show_trace=false,
#     method=TwoPointSteepestDescent(),
#     kwargs...,
# )
#     Threads.@threads for fname in files
#         # Deal with different types of inputs
#         if endswith(fname, ".res")
#             cell = read_res(fname)
#             label = cell.metadata[:label]
#         elseif endswith(fname, ".cell")
#             cell = read_cell(fname)
#             label = stem(fname)
#         end
#         calc =
#             EDDP.NNCalc(cell, cf, deepcopy(ensemble); nmax, core=CoreReplusion(core_size))
#         vc, _ = relax!(
#             calc;
#             relax_cell,
#             pressure_gpa,
#             show_trace,
#             method,
#             out_label=label,
#             kwargs...,
#         )
#         outname = joinpath(outdir, stem(fname) * ".res")
#         write_res(outname, get_cell(vc))
#     end
# end

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
        p = pressure_gpa / 160.21766208
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



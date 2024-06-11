using LinearAlgebra
using GarishPrint
#=
Relaxation of a structure
=#

using Printf
using Optim: optimize, LBFGS, BFGS
using Optim.LineSearches: BackTracking
using Configurations

@option mutable struct RelaxOption <: EDDPotentialsOption
    method::String = "tpsd"
    energy_threshold::Float64 = 1e-5
    force_threshold::Float64 = 1e-2
    stress_threshold::Float64 = 1e-1
    external_pressure::Vector{Float64} = [0.0]
    relax_cell::Bool = true
    keep_trajectory::Bool = false
    verbose::Bool = false
    iterations::Int = 1000
end

struct Relax{T}
    calc::T
    options::RelaxOption
    trajectory::Vector{Any}
end


Relax(calc, options=RelaxOption()) = Relax(calc, options, [])

function Base.show(io::IO, text::MIME"text/plain", x::Relax)
    println(io, "Relax:")
    println(io, "Calc:")
    show(io, text, x.calc)
    println(io, "Options:")
    if isa(io, GarishPrint.GarishIO)
        pprint_struct(GarishPrint.GarishIO(io; include_defaults=true), text, x.options)
    else
        pprint_struct(
            GarishPrint.GarishIO(io; include_defaults=true, color=false),
            text,
            x.options,
        )
    end
end


struct RelaxResult
    relax::Relax
    dE::Float64
    fmax::Float64
    smax::Float64
    iterations::Int
    converged::Bool
end

function Base.show(io::IO, text::MIME"text/plain", x::RelaxResult)
    pprint_struct(GarishPrint.GarishIO(io; color=false, include_defaults=true), text, x)
end

precompile(Base.show, (Base.TTY, MIME"text/plain", RelaxResult))
precompile(Base.show, (Base.TTY, MIME"text/plain", Relax))

"""
    stress_matrix(vector=AbstractVector)

Build a external stress matrix from vector inputs.
"""
function external_stress_matrix(vector::AbstractVector)
    @assert length(vector) in [6, 1] "Expect a vector with length 6 or 1"
    if length(vector) == 1
        a = vector[1]
        return diagm([a, a, a])
    elseif length(vector) == 6
        a, b, c, d, e, f = vector
        return [
            a b c
            b d e
            c e f
        ]
    end
end
external_pressure_matrix(x::Real) = external_pressure_matrix([x])


"""
    multirelax!(re::Relax;itermax=2000, restartmax=5)

Perform relaxation and restart if it does not converge. This is useful when the
algorithm suffer from vanishing step sizes and a reset of the state of the optimiser
helps.
"""
function multirelax!(re::Relax; max_iter=2000, max_restart=5)
    local outcome
    itertotal = 0
    for _ = 1:max_restart
        outcome = relax!(re)
        itertotal += outcome.iterations
        if outcome.converged || itertotal >= max_iter
            break
        end
    end
    RelaxResult(
        outcome.relax,
        outcome.dE,
        outcome.fmax,
        outcome.smax,
        itertotal,
        outcome.converged,
    )
end


"""
    relax!(re::Relax)

Perform relaxation for a `Relax` object.
"""
function relax!(re::Relax)
    (; method, energy_threshold, force_threshold, stress_threshold) = re.options
    (; external_pressure, relax_cell, keep_trajectory, verbose) = re.options
    (; iterations) = re.options
    trajectory = re.trajectory
    calc = re.calc

    if method == "tpsd"
        _method = TwoPointSteepestDescent()
    elseif method == "lbfgs"
        _method = LBFGS(; linesearch=BackTracking(maxstep=0.2))
    elseif method == "bfgs"
        _method = BFGS(; linesearch=BackTracking(maxstep=0.2))
    elseif method == "cg"
        _method = ConjugateGradient(; linesearch=BackTracking(maxstep=0.2))
    else
        throw(KeyError("Unknown `method`: $(method)"))
    end

    smat = external_stress_matrix(external_pressure)

    # Initial position vector

    if relax_cell
        _calc = VariableCellCalc(calc; external_pressure=smat)
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

    smax = 0.0
    fmax = 0.0
    de = 0.0
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
        Optim.Options(; callback, iterations);
        inplace=false,
    )

    if !relax_cell
        smax = -1.0
    end


    RelaxResult(re, de, fmax, smax, res.iterations, converged)
end



"""
    relax!(calc::AbstractCalc)

Optimise the cell using the Optim interface. Collect the trajectory if requested.
Note that the trajectory is collected for all force evaluations and may not 
corresponds to the actual iterations of the underlying LBFGS iterations.
"""
function relax!(calc::AbstractCalc; multi=true, relax_cell=true, method="tpsd", kwargs...)
    relax = Relax(calc, RelaxOption(; method, relax_cell, kwargs...))
    if multi
        multirelax!(relax)
    else
        relax!(relax)
    end
end

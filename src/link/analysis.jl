#=
Analysis and reporting functions for Builder workflows.

This module contains:
- Walk-forward testing for model validation
- Project status summaries
- Convenience methods for EDDP calculations
- Model structure generation
=#


raw"""
    walk_forward_tests(builder::Builder)

Perform walk forward tests - test if the data of generation ``N`` can be predicted by the model
from generation ``N-1``, and compare it with the results using model from generation ``N`` itself.

# Arguments
- `bu`: Builder object
- `print_results`: Print test results to stdout
- `iters`: Iterations to test (default: 0 to current-1)
- `fc_show_progress`: Show progress during feature loading
- `check_training_data`: Only test iterations with complete training data

# Returns
- Array of TrainingResults objects for each tested iteration
"""
function walk_forward_tests(
    bu::Builder;
    print_results=false,
    iters=0:bu.state.iteration-1,
    fc_show_progress=false,
    check_training_data=false,
)
    trs = []
    for iter in iters
        if check_training_data
            is_training_data_ready(bu, iter + 1) || continue
        else
            if nstructures_calculated(bu, iter + 1) < 1
                continue
            end
        end
        has_ensemble(bu, iter) || break
        @info "Loading features of generation $(iter+1) to test for generation $(iter)..."
        fc = load_features(bu, iter + 1; show_progress=fc_show_progress)
        ensemble = load_ensemble(bu, iter)
        tr = EDDPotentials.TrainingResults(ensemble, fc)
        push!(trs, tr)
        if print_results
            println(
                "Trained model using iteration 0-$iter applied for iteration $(iter+1):",
            )
            println(tr)
        end
    end
    trs
end


"""
    summarise(builder::Builder)

Summarise the status of the project, including structure counts for each iteration.
"""
function summarise(builder::Builder)
    opts = builder.state
    subpath(x) = joinpath(opts.workdir, x)
    println("Working directory: $(opts.workdir)")
    println("Seed file: $(opts.seedfile)")

    function print_res_count(path)
        nfiles = length(glob_allow_abs(joinpath(path, "*.res")))
        println("  $(path): $(nfiles) structures")
        nfiles
    end

    total_dft = 0
    for i = 0:opts.max_iterations
        println("iteration $i")

        path = subpath("gen$i")
        print_res_count(path)

        path = subpath("gen$(i)-dft")
        total_dft += print_res_count(path)
    end
    println("Total training structures: $(total_dft)")
end


# Map for trainer names
TRAINER_NAME = Dict(TrainingOption => "locallm")


# Generate convenience methods for EDDP calculations using Builder
for func in [:eddp_energy, :eddp_forces, :eddp_pressure, :eddp_energy_std, :eddp_enthalpy]
    @eval begin
        function $func(
            cell::Cell,
            builder::Builder,
            gen::Int=_latest_ensemble_iteration(builder);
            kwargs...,
        )
            ensemble = load_ensemble(builder, gen)
            $func(VariableCellCalc(NNCalc(cell, builder.cf, ensemble)); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, builder::Builder, gen::Int=_latest_ensemble_iteration(builder);kwargs...)

        Convenient method for calling $(EDDPotentials.$func) using a Builder object.
        """ $func(cell::Cell, builder::Builder, gen::Int)
    end
end


"""
    write_model_structures(bu::Builder; iteration)

Generate and write model structures for training at specified iteration.

These are systematically generated structures that probe different chemical
environments (2-body, 3-body, 4-body) at various distances.

# Arguments
- `bu`: Builder object
- `iteration`: Iteration number (default: current iteration)
"""
function write_model_structures(bu::Builder; iteration=bu.state.iteration)
    target_dir = joinpath(bu.state.workdir, "gen$iteration")
    ensure_dir(target_dir)
    rmin = bu.state.core_size
    rmax = suggest_rcut(bu.cf)
    seedstem = split(stem(bu.state.seedfile), "-")[1]
    for order in [2, 3, 4]
        pairs, cells = model_structures(bu.cf.elements; order, rmin, rmax)
        for (pair, this_cells) in zip(pairs, cells)
            p = join(pair, "-")
            for (n, cell) in enumerate(this_cells)
                name = joinpath(target_dir, "$seedstem-$p-$n.res")
                write_res(name, cell)
            end
        end
    end
end

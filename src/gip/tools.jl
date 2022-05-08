#=
Various tool functions for workflow managements
=#

using JLD2

"""
generate_vc(cell::Cell, ensemble::ModelEnsemble, cf::CellFeature;copy_cell=true, rcut=suggest_rcut(cf), nmax=500)

Generate a VariableCellFiler that handles variable cell relaxation
"""
function generate_vc(cell::Cell, ensemble::ModelEnsemble, cf::CellFeature;copy_cell=true, rcut=suggest_rcut(cf), nmax=500)
    copy_cell && deepcopy(cell)
    cw = CellTools.CellWorkSpace(cell;cf, nmax, rcut)
    calc = CellTools.CellCalculator(cw, ensemble)
    CellTools.VariableLatticeFilter(calc)
end


function relax_structures(pattern::AbstractString, en_path::AbstractString, cf;energy_threshold=20., savepath="relaxed", skip_existing=true)

    ensemble = jldopen(en_path) do file
        file["ensemble"]
    end

    loaded = CellTools.load_structures(pattern, cf;energy_threshold)

    isdir(savepath) || mkdir(savepath)

    p = Progress(length(loaded.cells))
    n = length(loaded.cells)

    function do_work(i)
        fname = splitpath(loaded.fpath[i])[end]
        # Skip and existing file
        skip_existing && isfile(joinpath(savepath, fname)) && return

        vc = generate_vc(loaded.cells[i], ensemble, cf)

        try
            CellTools.optimise_cell!(vc)
        catch
            return
        end

        this_cell = CellTools.get_cell(vc)
        # Set metadata
        # TODO add spacegroup information from Spglib.jl
        this_cell.metadata[:enthalpy] = CellTools.get_energy(vc)
        this_cell.metadata[:volume] = volume(this_cell)
        this_cell.metadata[:pressure] = CellTools.get_pressure_gpa(vc.calculator)
        this_cell.metadata[:label] = fname
        CellTools.write_res(joinpath(savepath, fname), this_cell)
    end

    @info "Total number of structures: $n"
    Threads.@threads for i in 1:n
        do_work(i)
        next!(p)
    end
end


function train(patterns, outpath, 
              feature_opts::FeatureOptions=FeatureOptions(), 
              training_options::TrainingOptions=TrainingOptions();
              energy_threshold=20.,
              )

    files_ = [glob(pattern) for pattern in patterns]
    files = reduce(vcat, files_)

    featurespec = CellFeature(feature_opts)

    celldata = CellTools.load_structures(files, featurespec;energy_threshold)

    @info "Number of structures: $(length(celldata.cells))"
    @info "Number of features: $(CellTools.nfeatures(featurespec))"

    # Prepare training data
    traindata = CellTools.training_data(celldata);

    # Train the models
    output = CellTools.train_multi(traindata, outpath, training_options)

    # Save the ensemble model
    CellTools.create_ensemble(output.savefile)
end

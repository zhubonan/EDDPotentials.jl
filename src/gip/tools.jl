#=
Various tool functions for workflow managements
=#

"""
Generate a VariableCellFiler that handles variable cell relaxation
"""
function generate_vc(cell::Cell, ensemble::ModelEnsemble, cf::CellFeature;copy_cell=true, rcut=suggest_rcut(cf), nmax=500)
    copy_cell && deepcopy(cell)
    cw = CellTools.CellWorkSpace(cell;cf, nmax, rcut)
    calc = CellTools.CellCalculator(cw, ensemble)
    CellTools.VariableLatticeFilter(calc)
end


function relax_structures(folder, cf;energy_threshold, savepath="relaxed", skip_existing=true)

    ensemble = jldopen(ARGS[1]) do file
        file["ensemble"]
    end

    loaded = CellTools.load_structures("$(folder)/*.res", cf;energy_threshold)

    isdir(savepath) || mkdir(savepath)

    p = Progress(length(loaded.cells))
    n = length(loaded.cells)

    @info "Total number of structures: $n"

    Threads.@threads for i in 1:n

        fname = splitpath(loaded.fpath[i])[end]
        # Skip and existing file
        skip_existing && isfile(joinpath(savepath, fname)) && continue

        vc = get_vc(loaded.cells[i], ensemble, cf)

        try
            CellTools.optimise_cell!(vc)
        catch
            next!(p)
            continue
        end

        this_cell = CellTools.get_cell(vc)
        # Set metadata
        # TODO add spacegroup information from Spglib.jl
        this_cell.metadata[:enthalpy] = CellTools.get_energy(vc)
        this_cell.metadata[:volume] = volume(this_cell)
        this_cell.metadata[:pressure] = CellTools.get_pressure_gpa(vc.calculator)
        this_cell.metadata[:label] = fname
        CellTools.write_res(joinpath(savepath, fname), this_cell)
        next!(p)
    end
end

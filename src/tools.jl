#=
Various tool functions for workflow managements
=#

using CellBase: rattle!
import CellBase: write_res
using Base.Threads
using JLD2
using Dates
using UUIDs

"""
generate_vc(cell::Cell, ensemble::ModelEnsemble, cf::CellFeature;copy_cell=true, rcut=suggest_rcut(cf), nmax=500)

Generate a VariableCellFiler that handles variable cell relaxation
"""
function generate_vc(cell::Cell, ensemble::ModelEnsemble, cf::CellFeature;copy_cell=true, rcut=suggest_rcut(cf), nmax=500)
    copy_cell && deepcopy(cell)
    cw = CellWorkSpace(cell;cf, nmax, rcut)
    calc = CellCalculator(cw, ensemble)
    VariableLatticeFilter(calc)
end


function relax_structures(pattern::AbstractString, en_path::AbstractString, cf;energy_threshold=20., savepath="relaxed", skip_existing=true)

    ensemble = jldopen(en_path) do file
        file["ensemble"]
    end

    loaded = load_structures(pattern, cf;energy_threshold)

    isdir(savepath) || mkdir(savepath)

    p = Progress(length(loaded.cells))
    n = length(loaded.cells)

    function do_work(i)
        fname = splitpath(loaded.fpath[i])[end]
        # Skip and existing file
        skip_existing && isfile(joinpath(savepath, fname)) && return

        vc = generate_vc(loaded.cells[i], ensemble, cf)

        try
            optimise_cell!(vc)
        catch
            return
        end
        write_res(joinpath(savepath, fname), vc;label=fname, symprec=0.1)
   end

    @info "Total number of structures: $n"
    Threads.@threads for i in 1:n
        do_work(i)
        next!(p)
    end
end


function train(patterns, outpath, 
              feature_opts::FeatureOptions, 
              training_options::TrainingOptions=TrainingOptions();
              energy_threshold=20. 
              )

    files_ = [glob(pattern) for pattern in patterns]
    files = reduce(vcat, files_)

    featurespec = CellFeature(feature_opts)

    celldata = load_structures(files, featurespec;energy_threshold)

    @info "Number of structures: $(length(celldata.cells))"
    @info "Number of features: $(nfeatures(featurespec))"

    # Prepare training data
    traindata = training_data(celldata);

    # Train the models
    output = train_multi_distributed(traindata, outpath, training_options;featurespec)

    # Save the ensemble model
    create_ensemble(output.savefile)
end

"""
    update_metadata!(vc::VariableLatticeFilter, label;symprec=1e-2)

Update the metadata attached to a `Cell`` object
"""
function update_metadata!(vc::VariableLatticeFilter, label;symprec=1e-2)
    this_cell = get_cell(vc)
    # Set metadata
    this_cell.metadata[:enthalpy] = get_energy(vc)
    this_cell.metadata[:volume] = volume(this_cell)
    this_cell.metadata[:pressure] = get_pressure_gpa(vc.calculator)
    this_cell.metadata[:label] = label
    symm = CellBase.get_international(this_cell, symprec)
    this_cell.metadata[:symm] = "($(symm))"
    # Write to the file
    vc
end

"""
    write_res(path, vc::VariableLatticeFilter;symprec=1e-2, label="EDDP")

Write structure in VariableCellFiler as SHELX file.
"""
function write_res(path, vc::VariableLatticeFilter;symprec=1e-2, label="EDDP")
    update_metadata!(vc, label;symprec)
    write_res(path, get_cell(vc))
end

"""
    build_and_relax(seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10)

Build the structure and relax it

"""
function build_and_relax(seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10)
    lines = open(seedfile, "r") do seed 
        cellout = read(pipeline(`timeout $(timeout) buildcell`, stdin=seed, stderr=devnull), String)
        split(cellout, "\n")
    end

    # Generate a unique label
    label = get_label(stem(seedfile))

    cell = read_cell(lines)
    vc = generate_vc(cell, ensemble, cf)
    optimise_cell!(vc)
    update_metadata!(vc, label)
    outpath = joinpath(outdir, "$(label).res")

    # Write out SHELX file
    write_res(outpath, get_cell(vc))
end

"""
    build_and_relax_one(seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10, warn=true)

Build and relax a single structure, ensure that the process *does* generate a new structure.
"""
function build_and_relax_one(seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10, warn=true, max_attempts=999)
    not_ok = true
    n = 1
    while not_ok && n <= max_attempts
        try
            build_and_relax(seedfile, outdir, ensemble, cf;timeout)
        catch err 
            if !isa(err, InterruptException)
                if warn
                    if typeof(err) <: ProcessFailedException 
                        println(stderr, "WARNING: `buildcell` failed to make the structure")
                    else
                        println(stderr, "WARNING: relaxation errored!")
                        println(stderr, "Error: $err")
                    end
                end
            else
                throw(err)
            end
            n += 1
            continue
        end
        not_ok=false
    end
end

"""
    build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10)

Build and relax `num` structures in parallel (threads) using passed `ModuleEnsemble` and `CellFeature`
"""
function build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10)
    pbar = Progress(num;desc="Build and relax: ")
    for i in 1:num
        build_and_relax_one(seedfile, outdir, ensemble, cf;timeout,warn=true)
        next!(pbar)
    end
end

"""
    build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble_file::AbstractString;timeout=10)

Build the structure and relax it
"""
function build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble_file::AbstractString;timeout=10)
    ensemble = load_ensemble_model(ensemble_file)
    featurespec = load_featurespec(ensemble_file)
    build_and_relax(num, seedfile, outdir, ensemble, featurespec;timeout)
end


ensure_dir(path) =  isdir(path) || mkdir(path)

function get_label(seedname)
    dt = Dates.format(now(), "yy-mm-dd-HH-MM-SS")
    suffix = string(uuid4())[end-7:end]
    "$(seedname)-$(dt)-$(suffix)"
end

stem(x) = splitext(splitpath(x)[end])[1]

"""
Call `buildcell` to generate many random structure under `outdir`
"""
function build_cells(seedfile, outdir, num;save_as_res=true, build_timeout=5, ntasks=nthreads())
    asyncmap((x) -> build_one_cell(seedfile, outdir;save_as_res, build_timeout), 1:num;
             ntasks=ntasks)
end

"""
Call `buildcell` to generate many random structure under `outdir`
"""
function build_one_cell(seedfile, outdir;save_as_res=true, build_timeout=5, suppress_stderr=false, max_attemps=999)
    not_ok = true
    suppress_stderr ? stderr_dst = devnull : stderr_dst=nothing
    n = 1
    while not_ok && n <= max_attemps
        outname = save_as_res ? get_label(stem(seedfile)) * ".res" : get_label(stem(seedfile)) * ".cell"
        outpath = joinpath(outdir, outname)
        try
            if save_as_res
                pip = pipeline(pipeline(`timeout $(build_timeout) buildcell`, stdin=seedfile), `timeout $(build_timeout) cabal cell res`)
                pip = pipeline(pip, stdout=outpath, stderr=stderr_dst)
            else
                pip = pipeline(`timeout $(build_timeout) buildcell`, stdin=seedfile, stdout=outpath, stderr=stderr_dst)
            end

            run(pip)
        catch err
            if typeof(err) <: ProcessFailedException
                rm(outpath)
                n += 1
                continue
            else
                throw(err)
            end
        end
        # Success
        not_ok = false
    end
end


function run_pp3_many(workdir, indir, outdir, seedfile;n_parallel=1, keep=false)
    files = glob(joinpath(indir, "*.res"))
    ensure_dir(workdir)
    for file in files
        working_path = joinpath(workdir, splitpath(file)[end])
        cp(file, working_path, force=true)
        try
            run_pp3(working_path, seedfile, joinpath(outdir, splitpath(file)[end]))
        catch error
            if typeof(error) <: ArgumentError
                @warn "Failed to calculate energy for $(file)!"
                continue
            end
            throw(error)
        end
        if !keep
            for suffix in [".cell", ".conv", "-out.cell", ".pp", ".res"]
                rm(swapext(working_path, suffix)) 
            end
        end
    end
end


function run_pp3(file::AbstractString, seedfile::AbstractString, outpath::AbstractString)
    if endswith(file, ".res")
        cell = CellBase.read_res(file)
        # Write as cell file
        CellBase.write_cell(swapext(file, ".cell"), cell)
    else
        cell = CellBase.read_cell(file)
    end
    # Copy the seed file
    cp(swapext(seedfile, ".pp"), swapext(file, ".pp"), force=true)
    # Run pp3 relax
    # Read enthalpy
    enthalpy=0.
    pressure=0.
    for line in eachline(pipeline(`pp3 -n $(splitext(file)[1])`))
        if contains(line, "Enthalpy")
            enthalpy = parse(Float64, split(line)[end])
        end
        if contains(line, "Pressure")
            pressure = parse(Float64, split(line)[end])
        end
    end 
    # Write res
    cell.metadata[:enthalpy] = enthalpy
    cell.metadata[:pressure] = pressure
    cell.metadata[:label] = stem(file)
    CellBase.write_res(outpath, cell)
    cell
end

swapext(fname, new) = splitext(fname)[1] * new



"""
    run_crud(workdir, indir, outdir;n_parallel=1, mpinp=4)

Use `crud.pl` to calculate energies of all files in the input folder and store
the results to the output folder.
It is assumed that the files are named like `SEED-XX-XX-XX.res` and the parameters
for calculations are stored under `<workdir>/SEED.cell` and `<workdir>/SEED.param`. 
"""
function run_crud(workdir, indir, outdir;n_parallel=1, mpinp=4)
    hopper_folder =joinpath(workdir, "hopper") 
    gd_folder =joinpath(workdir, "good_castep") 
    ensure_dir(hopper_folder)
    ensure_dir(outdir)

    # Clean the hopper folder
    rm.(glob(joinpath(hopper_folder, "*.res")))

    infiles = glob(joinpath(indir, "*.res"))
    existing_file_names = map(stem, glob(joinpath(outdir, "*.res")))

    # Copy files to the hopper folder, skipping any existing files in the output folder
    nfiles = 0
    for file in infiles
        stem(file) in existing_file_names && continue
        cp(file, joinpath(hopper_folder, stem(file) * ".res"), force=true)
        nfiles += 1
    end
    @info "Number of files to calculate: $(nfiles)"

    # Run n_parallel instances of crud.pl
    # TODO: This is only a temporary solution - should implement crud-like 
    # tool in Julia itself
    tasks = []
    try
        @sync begin
            for i=1:n_parallel
                push!(tasks, @async run(setenv(`crud.pl -singlepoint -mpinp $mpinp`, dir=workdir)))
                sleep(0.01)
            end
        end
    catch error
        if typeof(error) <: InterruptException
            Base.throwto.(tasks, InterruptException())
        end
        throw(error)
    end

    # Transfer files to the target folder
    nfiles = 0
    for file in infiles
        fname = stem(file) * ".res"
        fsrc = joinpath(gd_folder, fname)
        fdst = joinpath(outdir, fname)
        if isfile(fsrc)
            cp(fsrc, fdst, force=true)  
            rm(fsrc)
            nfiles += 1
        end

        # Copy CASTEP files is there is any
        fname = stem(file) * ".castep"
        fsrc = joinpath(gd_folder, fname)
        fdst = joinpath(outdir, fname)
        if isfile(fsrc)
            cp(fsrc, fdst, force=true)
            rm(fsrc)
        end
    end
    @info "Number of new structures calculated: $(nfiles)"
end

"""
    shake_res(files::Vector, nshake::Int, amp::Real)

Shake the given structures and write new files with suffix `-shake-N.res`.

"""
function shake_res(files::Vector, nshake::Int, amp::Real, cellamp::Real=0.02)
    for f in files
        cell = read_res(f)
        pos_backup = get_positions(cell)
        cellmat_backup = get_cellmat(cell)
        label = cell.metadata[:label]
        for i in 1:nshake
            rattle!(cell, amp)
            rattle_cell!(cell, cellamp)
            cell.metadata[:label] = label * "-shake-$i"
            write_res(splitext(f)[1] * "-shake-$i.res", cell)
            # Reset the original cellmatrix and positions
            set_cellmat!(cell, cellmat_backup)
            set_positions!(cell, pos_backup)
        end
    end
end

const train_eddp = train
export train_eddp


"""
    rattle_cell(cell::Cell, amp::Real)

Rattle the cell shape based on random fractional changes on the cell parameters.
"""
function rattle_cell!(cell::Cell, amp::Real)
    local new_cellpar
    i = 0
    while true
        new_cellpar = [x * (1 + rand()*amp) for x in cellpar(cell)]
        CellBase.isvalidcellpar(new_cellpar...) && break
        # Cannot found a valid cell parameters?
        if i > 10
            return cell
        end 
        i += 1
    end
    new_lattice = Lattice(new_cellpar)
    spos = get_scaled_positions(cell)
    CellBase.set_cellmat!(cell, cellmat(new_lattice))
    positions(cell) .= cellmat(cell) * spos
    cell
end
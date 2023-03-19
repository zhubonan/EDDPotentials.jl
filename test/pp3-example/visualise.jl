using EDDP
using CellBase
using Plots
link_file = joinpath(@__DIR__, "link.toml")

builder = Builder(link_file)
link!(builder)

cell = Cell(Lattice([10, 10, 10, 90, 90, 90.0]), [:Al, :Al], [0 1.0; 0 0; 0 0])
x = LinRange(1, 5, 100)

function E(x, calc)
    get_cell(calc).positions[1, 2] = x
    get_energy(calc)
end

begin
    images = []
    for i in x
        new_cell = deepcopy(cell)
        new_cell.positions[1, 2] = i
        push!(images, new_cell)
    end
    workdir = joinpath(builder.state.workdir, "lj-pp3")
    isdir(workdir) || mkdir(workdir)
    pp3_out = []
    bdir(x) = joinpath(builder.state.workdir, x)
    for (i, img) in enumerate(images)
        CellBase.write_cell(bdir("lj-pp3/lj-$(i).cell"), img)
        outcell = EDDP.run_pp3(bdir("lj-pp3/lj-$(i).cell"), bdir("Al"), nothing)
        push!(pp3_out, outcell.metadata[:enthalpy])
    end
end

"""
Scattering plot of the predicted and the actual energy
"""
function show_plot(builder, i)
    ensemble = load_ensemble(builder, i)
    fc = load_features(builder, 0:i)
    tr = TrainingResults(ensemble, fc)
    x = tr.H_pred ./ natoms(fc)
    y = tr.H_target ./ natoms(fc)
    scatter(x, y, xlim=(-10, 0), ylim=(-10, 0), aspect_ratio=:equal)
end

# Compare with PP results
"""
    plot_lj(builder, x=LinRange(1,5, 100))

Plot the energy from the fitted model and the LJ ground truth.
"""
function plot_lj(builder, x=LinRange(1, 5, 100))
    p = plot(
        x,
        pp3_out,
        label="LJ",
        xlim=(1.5, 5),
        ylim=(-3, 10),
        xlabel="Distance",
        ylabel="Energy",
        linewidth=5,
    )
    for iter = 0:builder.state.iteration
        EDDP.has_ensemble(builder, iter) || break
        ensemble = load_ensemble(builder, iter)
        calc = NNCalc(cell, builder.cf, ensemble)
        y = E.(x, Ref(calc))
        plot!(p, x, y, label="Iteration $iter")
    end
    p
end

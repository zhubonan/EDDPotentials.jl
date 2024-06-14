#=
Finite diff related routine
=#

using CellBase
using Base.Iterators
using EDDPotentials:
    GradientWorkspace, compute_fv_gv_one!, nbodyfeatures, CellFeature, nfeatures
using NLSolversBase: OnceDifferentiable, jacobian!

function get_testcell_h(supercell=(1, 1, 1))
    lattice = [
        10.0 0 0
        0 10.0 0
        0 0 10.0
    ]

    xs = 0:2.5:7.5
    ys = 0:2.5:7.5
    zs = 0:2.5:7.5
    pos = hcat(map(collect, Iterators.product(xs, ys, zs))...)

    testcell = Cell(Lattice(lattice), repeat([:H], size(pos, 2)), pos)
    if supercell != (1, 1, 1)
        testcell = CellBase.make_supercell(testcell, supercell...)
    end

    cf = CellFeature([:H])
    nf = nfeatures(cf)
    fvec = zeros(nf, length(testcell))
    testcell, cf, fvec
end

function get_testcell_ternary(supercell=(1, 1, 1))
    lattice = [
        10.0 0 0
        0 10.0 0
        0 0 10.0
    ]

    xs = 0:2.5:7.5
    ys = 0:2.5:7.5
    zs = 0:2.5:7.5
    pos = hcat(map(collect, Iterators.product(xs, ys, zs))...)[:, 1:2]
    symbols = vcat(
        repeat([:H], 1),
        repeat([:B], 1),
        # repeat([:O], 16), 
        # repeat([:S], 16)
    )
    testcell = Cell(Lattice(lattice), symbols, pos)
    if supercell != (1, 1, 1)
        testcell = CellBase.make_supercell(testcell, supercell...)
    end
    cf = CellFeature([:H, :B])
    nf = nfeatures(cf)
    fvec = zeros(nf, length(testcell))
    testcell, cf, fvec
end




"""
    feature_gradient_fd(cell, cf, iat)

Compute gradients via finite displacement
"""
function feature_grad_fd(cell, cf, iat)
    nf = nfeatures(cf)
    fvec = zeros(nf, length(cell))
    wk = GradientWorkspace(fvec; do_grad=false)

    function f(pos)
        pos_old = cell.positions[:, iat]
        cell.positions[:, iat] .= pos
        nl = NeighbourList(cell, 7, 500; savevec=true)
        output = stack(
            iat ->
                compute_fv_gv_one!(
                    wk,
                    cf.two_body,
                    cf.three_body,
                    iat,
                    cell,
                    nl;
                    offset=nbodyfeatures(cf, 1),
                ).fvec |> copy,
            1:length(cell),
            dims=2,
        )
        cell.positions[:, iat] = pos_old
        output
    end

    p0 = positions(cell)[:, iat]
    od = OnceDifferentiable(f, p0, f(p0); inplace=false)
    transpose(jacobian!(od, p0))
end

"""
     feature_grad(cell, cf, iat)
     
Compute the gradient of the feature vectors directly.
"""
function feature_grad(cell, cf, iat)
    nf = nfeatures(cf)
    fvec = zeros(nf, length(cell))
    wk = GradientWorkspace(fvec; do_grad=true)
    nl = NeighbourList(cell, 7, 500; savevec=true)
    compute_fv_gv_one!(
        wk,
        cf.two_body,
        cf.three_body,
        iat,
        cell,
        nl;
        offset=nbodyfeatures(cf, 1),
    )
    wk
end

"""
    max_grad_dev(cell, cf)

Maximum deviation comparing finite diff and direct feature vector gradient calculation
"""
function max_grad_dev(cell, cf, iat)
    nf = nfeatures(cf)
    jac = reshape(feature_grad_fd(cell, cf, iat), 3, nf, length(cell))
    wk = feature_grad(cell, cf, iat)
    # Maximum diff 
    diffs = map(1:length(cell)) do i
        idx = findfirst(x -> x == i, wk.gvec_index.index)
        if idx == nothing
            return maximum(abs.(jac[:, :, i]))
        end
        maximum(abs.(wk.gvec[:, :, idx] .+ jac[:, :, i]))
    end
    maximum(diffs), wk, jac
end

cell, cf, fvec = get_testcell_ternary()

maxd, wk, jac = max_grad_dev(cell, cf, 1)

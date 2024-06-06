using EDDP
using CellBase
using Test


function _h2o_cell(l=4.0, factor=1.0)
    tmp = Float64[
        0   0.1  1.
        0   1.0  1.
        0.1 0.0  1.
    ] .* factor
    Cell(Lattice(l, l, l), [:H, :H, :O], tmp)
end

function _lco_cell()
    scaled_pos = Float64[

   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.6666666666666666    0.3333333333333333    0.3333333333333333 
   0.3333333333333333    0.6666666666666666    0.6666666666666666
   0.3333333333333333    0.6666666666666665    0.1666666666666667
   0.9999999999999999    0.9999999999999998    0.5000000000000000
   0.6666666666666665    0.3333333333333330    0.8333333333333333
   0.0000000000000000    0.0000000000000000    0.2400068000000000
   0.6666666666666666    0.3333333333333333    0.0933265333333333
   0.6666666666666666    0.3333333333333333    0.5733401333333333
   0.3333333333333333    0.6666666666666666    0.4266598666666666
   0.3333333333333333    0.6666666666666666    0.9066734666666667
   0.0000000000000000    0.0000000000000000    0.7599931999999999
    ] 
    cell = Float64[
           1.4056284883992509   -2.4346199584737427    0.0000000000000000
   1.4056284883992509    2.4346199584737427    0.0000000000000000
   0.0000000000000000    0.0000000000000000   13.9094564325679286
    ]
    cell = collect(transpose(cell))
    pos = cell * transpose(scaled_pos)

    Cell(Lattice(cell), [:Li, :Li, :Li, :Co, :Co, :Co, :O, :O, :O, :O, :O, :O], pos)
end

"""
    fd_gradient(cf, cell)

    Compute the finite difference gradient of the features with respect to the positions of the atoms.
Return the gradient of the feature indexed by [dir, nf, j, i], where i is tha atom that is moved and
j is the gradient of the feature vector of the j atom as a result of the movement of i.
"""
function fd_gradient(cf, cell)

    fb = EDDP.compute_fv_gv(cf, cell)
    gvec0 = copy(fb.gvec) # dFi/drj order

    # Recover the gradient in the dFj/dri format
    gvec_ideal = zeros(size(gvec0, 1), size(gvec0, 2), size(gvec0, 4), size(gvec0, 4))
    for iat in axes(gvec0, 4)
        for j in 1:fb.gvec_nn[iat]
            # Transfor neighbour local index to atom index
            jat = fb.gvec_index[j, iat]  # index of the atoms that has moved
            gvec_ideal[:, :, iat, jat] .+= gvec0[:, :, j, iat]
        end
    end

    diff = zeros(size(gvec0, 1), size(gvec0, 2), natoms(cell), natoms(cell))  # dFj/dri
    for iat in 1:natoms(cell)
        for dir in 1:3  
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += 1e-6
            fb = EDDP.compute_fv_gv(cf, dcell)
            fv1 = copy(fb.fvec)

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= 1e-6
            fb = EDDP.compute_fv_gv(cf, dcell)
            fv2 = copy(fb.fvec)
            diff[dir, :, :, iat] .= (fv1 - fv2) / 2e-6
        end
    end

    return diff, gvec_ideal

end


# Test with final force creation
"""
    force_gradient(cf, cell)

    Finite difference gradient of the energies with respect to the positions of the atoms.
Returns the gradient from finite difference and analytical computation. A random linear model is
used for prediction.
"""
function force_gradient(cf, cell)
    fb = EDDP.compute_fv_gv(cf, cell)
    gvec0 = copy(fb.gvec) # dFi/drj order
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    EDDP._force_update!(fb, gv;offset=length(cf.elements))
    EDDP._stress_update!(fb, gv;offset=length(cf.elements))
    forces = copy(fb.forces)

    # dE/dri
    diff = zeros(size(gvec0, 1), natoms(cell)) 
    for iat in 1:natoms(cell)
        for dir in 1:3  
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += 1e-6
            fb = EDDP.compute_fv_gv(cf, dcell)
            # Compute the energy
            e1 =  sum(param * fb.fvec)

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= 1e-6
            fb = EDDP.compute_fv_gv(cf, dcell)
            e2 =  sum(param * fb.fvec)
            # Compute the energy
            diff[dir, iat] = (e1 - e2) / 2e-6
        end
    end
    return diff, forces
end



"""
    stress_gradient(cf, cell)

    Finite difference gradient of the energies with respect to cell deformations.
Returns the gradient from finite difference and analytical computation. A random linear model is
used for prediction.
"""
function stress_gradient(cf, cell)
    fb = EDDP.compute_fv_gv(cf, cell)
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    EDDP._force_update!(fb, gv;offset=length(cf.elements))
    EDDP._stress_update!(fb, gv;offset=length(cf.elements))
    stress = copy(fb.tot_stress)


    smat_orig = diagm([1., 1., 1.])

    # dE/dri
    diff = zeros(Float64, 3, 3)
    for i in 1:3
        for j in 1:3  
            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] += 1e-6
            set_cellmat!(dcell, smat * cellmat(dcell);scale_positions=true)
            fb = EDDP.compute_fv_gv(cf, dcell)
            # Compute the energy
            e1 =  sum(param * fb.fvec)

            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] -= 1e-6
            set_cellmat!(dcell, smat * cellmat(dcell);scale_positions=true)
            fb = EDDP.compute_fv_gv(cf, dcell)
            e2 =  sum(param * fb.fvec)
            # Compute the energy
            diff[i, j] = (e1 - e2) / 2e-6
        end
    end
    return diff, stress
end

@testset "Gradients" begin

    # Run test cases
    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, gvec_ji = fd_gradient(cf, cell);
    @test maximum(abs.(diff - gvec_ji)) < 1e-6

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, gvec_ji = fd_gradient(cf, cell);
    @test maximum(abs.(diff - gvec_ji)) < 1e-6

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, forces = force_gradient(cf, cell);
    @test maximum(abs.(diff + forces)) < 1e-6

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, forces = force_gradient(cf, cell);
    @test maximum(abs.(diff + forces)) < 1e-6

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, varial = stress_gradient(cf, cell);
    @test maximum(abs.(diff + varial))  < 1e-5

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, varial = stress_gradient(cf, cell);
    @test maximum(abs.(diff + varial)) < 1e-5

end
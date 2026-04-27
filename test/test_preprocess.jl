using EDDPotentials
using CellBase
using Test
using StatsBase
using Random
using LinearAlgebra: norm

@testset "Preprocess" begin
    fpath = joinpath(datadir, "training/*.res")
    fpath = relpath(fpath, pwd())
    sc = EDDPotentials.StructureContainer([fpath])

    # Test indexing
    @test isa(sc[1], Cell)
    @test isa(sc[1:2], EDDPotentials.StructureContainer)
    @test length(sc) == 11
    labels = [x.metadata[:label] for x in sc.structures]
    @test isa(sc[[labels[1], labels[2]]], EDDPotentials.StructureContainer)
    @test isa(collect(sc)[1], CellBase.Cell)

    # test splitting
    sc1, sc2 = split(sc, 1, 2)
    @test length(sc1) == 1
    @test length(sc2) == 2

    sc3 = sc1 + sc2
    @test length(sc3) == length(sc1) + length(sc2)
    @test length(sc3) == 3

    sc1, sc2 = split(sc[1:10], 0.1, 0.2)
    @test length(sc1) == 1
    @test length(sc2) == 2

    sc_train, sc_val = EDDPotentials.train_val_split(sc, ratio_val=0.5)
    @test length(sc_train) == 6
    # Default
    fc = EDDPotentials.FeatureStore(sc)
    @test length(fc) == 11
    fc_train, fc_val = EDDPotentials.train_val_split(fc, ratio_val=0.5)
    @test length(fc_train) == 6

    @test isa(fc[[labels[1], labels[2]]], EDDPotentials.FeatureStore)
    @test isa(collect(fc)[1], Matrix)

    # Test standardisation — transforms are fitted in-place, fvecs are NOT mutated
    fc1, fc2 = split(fc[1:10], 5, 3; shuffle=false, standardize=false)
    fc11, fc12 = EDDPotentials.standardize!(fc1, fc2)
    # standardize! returns the same containers (mutated in-place)
    @test fc11 === fc1
    @test fc12 === fc2
    # fvecs should be unchanged (only transforms are stored, not applied to data)
    @test !isnothing(fc11.xt)
    @test !isnothing(fc11.yt)
    # Bug fix: verify second container uses the same transform from the first
    @test fc11.xt === fc12.xt
    @test fc11.yt === fc12.yt

    # Combine
    fc3 = fc1 + fc2
    @test length(fc3) == length(fc1) + length(fc2)

    # Original fc was not standardized
    @test fc.xt === nothing
    @test fc.yt === nothing
    # fvecs remain untransformed (standardize! only stores transforms, does not touch data)
    @test get_data(fc1)[1] ≈ get_data(fc)[1]

end

@testset "split_and_featurize" begin
    fpath = joinpath(datadir, "training/*.res")
    fpath = relpath(fpath, pwd())
    sc = EDDPotentials.StructureContainer([fpath])
    cf = CellFeature([:B]; p2=[2, 4], p3=[2], q3=[1], rcut2=4.0, rcut3=3.5)

    # Old path: featurize all then split
    fc_old = FeatureStore(sc, cf; show_progress=false)
    train_old, val_old, test_old =
        split(fc_old, 0.6, 0.2, 0.2; seed=42, standardize=true)

    # New path: split first then featurize
    train_new, val_new, test_new = EDDPotentials.split_and_featurize(
        sc,
        cf,
        0.6,
        0.2,
        0.2;
        seed=42,
        show_progress=false,
    )

    # Same split counts
    @test length(train_new) == length(train_old)
    @test length(val_new) == length(val_old)
    @test length(test_new) == length(test_old)

    # Feature vectors should be identical (same seed, same structures)
    for i in 1:length(train_new)
        @test train_new.data[i] ≈ train_old.data[i]
    end

    # Standardization transforms from train should match
    @test train_new.xt.mean ≈ train_old.xt.mean
    @test train_new.xt.scale ≈ train_old.xt.scale
    # val/test share the train transform
    @test val_new.xt === train_new.xt
    @test test_new.xt === train_new.xt

    # Without standardization
    fcs = EDDPotentials.split_and_featurize(
        sc,
        cf,
        5,
        3;
        seed=99,
        standardize=false,
        show_progress=false,
    )
    @test length(fcs) == 2
    @test fcs[1].xt === nothing
end

@testset "Synthetic force data" begin
    cell = _h2o_cell()
    forces = Float64[
        0.0 0.2 -0.1
        0.0 -0.3 0.4
        0.0 0.1 0.2
    ]
    cell.metadata[:force] = forces
    cell.metadata[:label] = "force-parent"

    sc = StructureContainer([cell], [-12.5], ["force-parent"])

    synth = synthetic_force_data(sc; nsynthetic=2, max_disp=0.05, rng=MersenneTwister(7))
    @test length(synth) == 2
    @test all(occursin.("-force-shake-", synth.paths))

    parent_pos = get_positions(cell)
    for (idx, newcell) in enumerate(synth.structures)
        disp = get_positions(newcell) .- parent_pos
        @test size(disp) == size(forces)
        @test maximum(norm.(eachcol(disp))) <= 0.05 + 1e-12
        @test newcell.metadata[:synthetic] === true
        @test newcell.metadata[:synthetic_parent_label] == "force-parent"
        @test newcell.metadata[:synthetic_parent_energy] ≈ -12.5
        @test newcell.metadata[:label] == synth.paths[idx]

        expected_delta = -sum(forces .* disp)
        @test synth.H[idx] ≈ -12.5 + expected_delta atol = 1e-12
        @test newcell.metadata[:synthetic_energy_delta] ≈ expected_delta atol = 1e-12
        @test newcell.metadata[:synthetic_displacement_norm] ≈ norm(disp) atol = 1e-12
        @test newcell.metadata[:synthetic_max_displacement] ≈
              maximum(norm.(eachcol(disp))) atol = 1e-12
    end

    # Zero-force atoms are still rattled under the shared-cap sampling rule.
    first_disp = get_positions(synth.structures[1]) .- parent_pos
    @test norm(first_disp[:, 1]) > 0.0

    synth_with_original = synthetic_force_data(
        sc;
        nsynthetic=1,
        max_disp=0.02,
        rng=MersenneTwister(9),
        keep_original=true,
    )
    @test length(synth_with_original) == 2
    @test synth_with_original.paths[1] == "force-parent"
    @test synth_with_original.H[1] ≈ -12.5

    cell2 = deepcopy(cell)
    set_positions!(cell2, get_positions(cell2) .+ 0.3)
    cell2.metadata[:force] = forces
    cell2.metadata[:label] = "force-parent-2"
    grouped_sc = StructureContainer([cell, cell2], [-12.5, -12.0], ["force-parent", "force-parent-2"])
    grouped_synth = synthetic_force_data(
        grouped_sc;
        nsynthetic=2,
        max_disp=0.03,
        rng=MersenneTwister(12),
    )
    split_a, split_b = split(
        grouped_synth,
        1,
        1;
        shuffle=false,
        respect_shakes=true,
        standardize=false,
    )
    @test length(split_a) == 2
    @test length(split_b) == 2

    cf = CellFeature(unique(species(cell)); p2=[2, 3], p3=[2], q3=[1], rcut2=4.0, rcut3=3.5)
    fc = FeatureStore(synth, cf; show_progress=false)
    @test length(fc) == 2
    @test fc.H ≈ synth.H

    missing_force_cell = deepcopy(cell)
    delete!(missing_force_cell.metadata, :force)
    @test_throws ArgumentError synthetic_force_data(
        StructureContainer([missing_force_cell], [-12.5], ["missing-force"]);
        nsynthetic=1,
        max_disp=0.05,
        rng=MersenneTwister(1),
    )

    bad_cell = deepcopy(cell)
    bad_cell.metadata[:force] = zeros(2, length(bad_cell))
    @test_throws ArgumentError synthetic_force_data(
        StructureContainer([bad_cell], [-12.5], ["bad-force"]);
        nsynthetic=1,
        max_disp=0.05,
        rng=MersenneTwister(1),
    )
end

@testset "StructureContainer extxyz" begin
    mktempdir() do tmpdir
        extxyz_path = joinpath(tmpdir, "structures.extxyz")
        write(
            extxyz_path,
            """
            2
            Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.5 name=frame-a pbc="T T T"
            H 0.0 0.0 0.0 0.0 0.0 0.1
            H 0.0 0.0 0.8 0.0 0.0 -0.1
            2
            Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.2 pbc="T T T"
            H 0.1 0.0 0.0 0.2 0.0 0.0
            H 0.0 0.2 0.8 -0.2 0.0 0.0
            """,
        )

        sc = StructureContainer(extxyz_path)
        @test length(sc) == 2
        @test sc.paths[1] == "frame-a"
        @test sc.paths[2] == "structures-2"
        @test sc.H[1] ≈ -1.5
        @test sc.H[2] ≈ -1.2
        @test sc.structures[1].metadata[:enthalpy] ≈ -1.5
        @test sc.structures[2].metadata[:enthalpy] ≈ -1.2
        @test haskey(sc.structures[1].metadata, :force)
        @test size(sc.structures[1].metadata[:force]) == (3, 2)
        @test sc.structures[1].metadata[:volume] ≈ 125.0
    end
end

@testset "ExtXYZ @slice syntax" begin
    # Test helper functions directly
    @test EDDPotentials._parse_path_slice("train.xyz@:2000") == ("train.xyz", ":2000")
    @test EDDPotentials._parse_path_slice("train.xyz@-1000:") == ("train.xyz", "-1000:")
    @test EDDPotentials._parse_path_slice("train.xyz@::2") == ("train.xyz", "::2")
    @test EDDPotentials._parse_path_slice("train.xyz@100:500:3") == ("train.xyz", "100:500:3")
    @test EDDPotentials._parse_path_slice("train.xyz") == ("train.xyz", nothing)
    @test EDDPotentials._parse_path_slice("path@something.xyz") == ("path@something.xyz", nothing)
    @test EDDPotentials._parse_path_slice("a@b.xyz@:10") == ("a@b.xyz", ":10")

    # Test _python_slice_to_range
    total = 20000
    @test EDDPotentials._python_slice_to_range(":2000", total) == 1:2000
    @test EDDPotentials._python_slice_to_range("-1000:", total) == 19001:20000
    @test EDDPotentials._python_slice_to_range("::2", total) == 1:2:20000
    @test EDDPotentials._python_slice_to_range("100:500:3", total) == 101:3:500
    @test EDDPotentials._python_slice_to_range(":", total) == 1:20000
    @test EDDPotentials._python_slice_to_range("::", total) == 1:20000
    @test EDDPotentials._python_slice_to_range("::1", total) == 1:20000
    @test EDDPotentials._python_slice_to_range("0:1", total) == 1:1
    @test EDDPotentials._python_slice_to_range("-1:", total) == 20000:20000
    @test EDDPotentials._python_slice_to_range(":-1", total) == 1:20000
    @test EDDPotentials._python_slice_to_range("5:5", total) == 6:5  # empty range, python [5:5] = []

    mktempdir() do tmpdir
        # Write a 4-frame extxyz file
        extxyz_path = joinpath(tmpdir, "multi.xyz")
        frames = ""
        for k in 1:4
            frames *= """
            2
            Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3 energy=-$(1.0+k*0.1) name=frame-$k pbc="T T T"
            H 0.0 0.0 0.0
            H 0.0 0.0 0.$k
            """
        end
        write(extxyz_path, frames)

        # Test _count_extxyz_frames
        @test EDDPotentials._count_extxyz_frames(extxyz_path) == 4

        # Full load (no slice)
        sc_all = StructureContainer(extxyz_path)
        @test length(sc_all) == 4

        # First 2 frames: @:2
        sc_first2 = StructureContainer("$(extxyz_path)@:2")
        @test length(sc_first2) == 2
        @test sc_first2.H[1] ≈ -1.1
        @test sc_first2.H[2] ≈ -1.2

        # Last 1 frame: @-1:
        sc_last = StructureContainer("$(extxyz_path)@-1:")
        @test length(sc_last) == 1
        @test sc_last.H[1] ≈ -1.4

        # Every 2nd frame: @::2
        sc_even = StructureContainer("$(extxyz_path)@::2")
        @test length(sc_even) == 2
        @test sc_even.H[1] ≈ -1.1
        @test sc_even.H[2] ≈ -1.3

        # Skip first: @1:
        sc_skip1 = StructureContainer("$(extxyz_path)@1:")
        @test length(sc_skip1) == 3
        @test sc_skip1.H[1] ≈ -1.2

        # Single frame: @0:1
        sc_one = StructureContainer("$(extxyz_path)@0:1")
        @test length(sc_one) == 1
        @test sc_one.H[1] ≈ -1.1
    end
end

using EDDPotentials
import EDDPotentials.Potential: get_cell, eddp_energy
using EDDPotentials.Search:
    _rss_candidate,
    _rss_select_pruned,
    _rss_pool_mean,
    _rss_pool_median,
    _rss_should_flush,
    _validate_rss_prune_option
using CellBase
using Test

struct _RssEnergyCalc{T} <: AbstractCalc
    cell::Cell{T}
    energy::Float64
end

get_cell(calc::_RssEnergyCalc) = calc.cell
eddp_energy(calc::_RssEnergyCalc; kwargs...) = calc.energy

function _rss_prune_test_cell(label, distance, enthalpy; symbols=[:H, :H])
    cell = Cell(
        Lattice(10.0, 10.0, 10.0),
        symbols,
        [0.0 distance; 0.0 0.0; 0.0 0.0],
    )
    cell.metadata[:label] = label
    cell.metadata[:enthalpy] = enthalpy
    return cell
end

@testset "RSS pruning" begin
    @testset "rank and deduplicate" begin
        opt = RssPruneOption(
            enabled=true,
            pool_size=4,
            keep_fraction=0.5,
            dedup_tol=1e-2,
            fingerprint_lim=5.0,
        )
        pool = _rss_candidate.([
            _rss_prune_test_cell("low-a", 1.0, -4.0),
            _rss_prune_test_cell("low-duplicate", 1.0, -3.0),
            _rss_prune_test_cell("next-unique", 2.0, -2.0),
            _rss_prune_test_cell("high-unique", 3.0, -1.0),
        ])

        selected, duplicates = _rss_select_pruned(pool, opt, 10)

        @test length(selected) == 2
        @test duplicates == 1
        @test [c.cell.metadata[:label] for c in selected] == ["low-a", "next-unique"]
    end

    @testset "flush triggers" begin
        size_opt = RssPruneOption(enabled=true, pool_size=3)
        pool = _rss_candidate.([
            _rss_prune_test_cell("a", 1.0, -3.0),
            _rss_prune_test_cell("b", 2.0, -2.0),
            _rss_prune_test_cell("c", 3.0, -1.0),
        ])
        stats = [(; mean=_rss_pool_mean(pool[1:i]),
                    median=_rss_pool_median(pool[1:i])) for i in eachindex(pool)]
        reason, should_flush = _rss_should_flush(pool, stats, size_opt)
        @test should_flush
        @test reason == :pool_size

        stable_opt = RssPruneOption(
            enabled=true,
            pool_size=100,
            min_stable_pool_size=3,
            stable_window=2,
            mean_abs_tol=1e-2,
            median_abs_tol=1e-2,
        )
        stable_pool = _rss_candidate.([
            _rss_prune_test_cell("s1", 1.0, -1.0000),
            _rss_prune_test_cell("s2", 2.0, -1.0001),
            _rss_prune_test_cell("s3", 3.0, -1.0002),
        ])
        stable_stats = [(; mean=_rss_pool_mean(stable_pool[1:i]),
                           median=_rss_pool_median(stable_pool[1:i])) for i in eachindex(stable_pool)]
        reason, should_flush = _rss_should_flush(stable_pool, stable_stats, stable_opt)
        @test should_flush
        @test reason == :stable_statistics

        selected, _ = _rss_select_pruned(stable_pool, stable_opt, 10)
        @test length(selected) == 1
    end

    @testset "option validation" begin
        @test _validate_rss_prune_option(RssPruneOption(enabled=true)) isa RssPruneOption
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, pool_size=0))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, keep_fraction=0.0))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, keep_fraction=NaN))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, dedup_tol=-1.0))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, fingerprint_lim=Inf))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, stable_window=-1))
        @test_throws ArgumentError _validate_rss_prune_option(RssPruneOption(enabled=true, mean_abs_tol=-1.0))
    end

    @testset "energy threshold" begin
        composition_minima = Dict{Composition,Float64}()
        low = _RssEnergyCalc(_rss_prune_test_cell("low", 1.0, -2.0), -2.0)
        high = _RssEnergyCalc(_rss_prune_test_cell("high", 2.0, 1.0), 1.0)

        @test EDDPotentials.check_energy_threshold!(composition_minima, low, 0.5)
        @test !EDDPotentials.check_energy_threshold!(composition_minima, high, 0.5)
    end
end

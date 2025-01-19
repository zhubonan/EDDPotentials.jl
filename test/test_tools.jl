using EDDPotentials: compute_rdf

@testset "rdf" begin
    cell = _h2o_cell()
    bins = 0.1:0.05:5
    counts = compute_rdf(cell; bins)
    @assert length(bins) == length(counts)
end
module EDDPCli

using Comonicon
using EDDP: link!, Builder, _run_rss, load_ensemble, ensure_dir

# write your code here
"""
    link

    Perform iterative building of the EDDP potential
"""
@cast function link(fname::String="link.yaml"; iter::Int=-1)
    builder = Builder(fname)
    if iter >= 0
        builder.state.iteration = iter
    end
    link!(builder)
end

"""
    run_rss

Perform random structure searching
"""
@cast function run_rss(fname::String="link.yaml";)
    builder = Builder(fname)
    ensemble = load_ensemble(builder, rs.ensemble_id)
    searchdir = joinpath(builder.state.workdir, rs.subfolder_name)
    ensure_dir(searchdir)
    _run_rss(
        rs.seedfile,
        ensemble,
        builder.cf;
        show_progress=rs.show_progress,
        max=rs.max,
        outdir=searchdir,
        ensemble_std_max=rs.ensemble_std_max,
        ensemble_std_min=rs.ensemble_std_min,
        packed=rs.packed,
        niggli_reduce_output=rs.niggli_reduce_output,
        max_err=rs.max_err,
        kwargs...,
    )
end



"""
Command line interface for EDDP.jl
"""
@main

end # EDDPCli.jl

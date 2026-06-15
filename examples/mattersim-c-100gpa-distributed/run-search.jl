using EDDPotentials

builder = Builder("link.toml")
builder.rss.max = 4
builder.rss.subfolder_name = "verify-search-100gpa"
builder.rss.pressure_gpa = 100.0
builder.rss.show_progress = false
builder.rss.packed = false

result = run_rss(builder)
outdir = joinpath(builder.state.workdir, builder.rss.subfolder_name)
if abs(builder.rss.pressure_gpa) > 0.01
    outdir *= "-$(builder.rss.pressure_gpa)gpa"
end
outdir = result === nothing ? outdir : result
files = sort(filter(endswith(".res"), readdir(outdir; join=true)))
println("verification_search_dir = ", outdir)
println("verification_search_res_count = ", length(files))
for file in files
    println(file)
end

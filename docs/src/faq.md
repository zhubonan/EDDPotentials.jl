# FAQ

## Building custom system image

At the time of writing, loading Julia modules comes with high latency in addition to the the so called time-to-first-X (TTTX) delay related to time spent on compiling native code in each fresh Julia session.

Such delays can be completely eliminated by compiling a system image using the [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl), which contain `EDDPotential.jl`, `EDDPotentialTools.jl` and all its dependencies. 
This can be done by simply following the documentation of the PackageCompiler.jl 

For development environment this is a bit more complicated while the inconvenience of compiler latency is more pronounce due to frequent session restarts.
In this case, a special system image needs to be created that includes all dependencies of `EDDPotential.jl` and `EDDPotentialTools.jl`, plus some commonly used tools (such as `BenchmarkTools.jl` and `Revise`), but not `EDDPotential.jl` and `EDDPotentialTools.jl` themselves. 
Create such system image, there is a `sysimg.jl` under the `scripts` folder that can used via:

```bash
julia --project=<some project with PackageCompiler installed> -e 'include("sysimg.jl");build()' 
```

This creates the system image file at `~/.julia/eddpdev.so`.
To launch julia with the system image:

```bash
julia -J ~/.julia/eddpdev.so
```

```julia-repl
julia> "Flux" in [x.name for x in keys(Base.loaded_modules)]
true
```

One should note that any packages compiled into the system image is essentially frozen-in - their version cannot be changed by `Pkg`.
Hence, the system image must be rebuilt after any update of the `Manifest.toml`.


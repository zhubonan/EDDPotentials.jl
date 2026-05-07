#=
Shared utility functions for the link module.

This module contains common utilities used across the link package:
- Directory management
- Path helpers
- UUID management
- Dictionary key conversion
=#


"""
    ensure_dir(args...)

Create directory if it doesn't exist. Joins path components and creates the directory.
"""
ensure_dir(args...) = isdir(joinpath(args...)) || mkdir(joinpath(args...))

"""
    input_structure_dir(workdir, iteration)

Get the directory path for input structures (before DFT calculations) for a given iteration.
Returns: `workdir/gen{iteration}`
"""
input_structure_dir(workdir, iteration) = joinpath(workdir, "gen$(iteration)")

"""
    output_structure_dir(workdir, iteration)

Get the directory path for output structures (after DFT calculations) for a given iteration.
Returns: `workdir/gen{iteration}-dft`
"""
output_structure_dir(workdir, iteration) = joinpath(workdir, "gen$(iteration)-dft")


"""
    builder_uuid(workdir='.')

Get or create a UUID for the builder project. The UUID is stored in `.eddp_builder`
file in the working directory and remains persistent across sessions.

# Arguments
- `workdir`: Working directory path (default: current directory)

# Returns
- UUID string
"""
function builder_uuid(workdir='.')
    fname = joinpath(workdir, ".eddp_builder")
    if isfile(fname)
        uuid = open(fname) do fh
            chomp(readline(fh))
        end
    else
        uuid = string(uuid4())
        open(fname, "w") do fh
            write(fh, uuid)
            write(fh, "\n")
        end
    end
    uuid
end

"""
    builder_short_uuid(workdir)

Get shortened (8 character) version of the builder UUID.

# Arguments
- `workdir`: Working directory path

# Returns
- First 8 characters of the UUID
"""
builder_short_uuid(workdir) = builder_uuid(workdir)[1:8]


"""
    make_symbol_keys(dict::Dict{T,K}) where {T,K}

Recursively convert dictionary keys from strings to symbols.

# Arguments
- `dict`: Dictionary with string keys

# Returns
- Dictionary with Symbol keys

# Example
```julia
d = Dict("a" => 1, "b" => Dict("c" => 2))
make_symbol_keys(d)  # Dict(:a => 1, :b => Dict(:c => 2))
```
"""
function make_symbol_keys(dict::Dict{T,K}) where {T,K}
    Dict{Symbol,K}(Symbol(key) => make_symbol_keys(value) for (key, value) in pairs(dict))
end

"""
    make_symbol_keys(x)

Identity function for non-dict values (recursion base case).
"""
make_symbol_keys(x) = x


"""
    add_threads_env(cmd; julia_threads=1, omp_threads=1)

Set JULIA_NUM_THREADS and OMP_NUM_THREADS environment variables on a command.

# Arguments
- `cmd`: Command to modify

# Keywords
- `julia_threads`: Number of Julia threads (default 1)
- `omp_threads`: Number of OpenMP threads (default 1)

# Returns
- Modified command with environment variables set
"""
add_threads_env(cmd; julia_threads=1, omp_threads=1) = addenv(cmd,
    "JULIA_NUM_THREADS" => julia_threads,
    "OMP_NUM_THREADS" => omp_threads,
)

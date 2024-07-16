#=
Code for fast loading SHELX dataset 

The idea is to not load the structure but only scan the text for the information
about the composition and the TITL line. 

Composition parsing takes a significant amount of time at the moment.
For even faster loading, the composition string should be encoded inside the REM information....

=#

using CellBase
import CellBase
using StatsBase
export read_shelx_record

"""
    ShelxTITL

Information from the TITL line of an AIRSS style SHELX file
"""
struct ShelxTITL
    label::String
    pressure::Float64
    volume::Float64
    enthalpy::Float64
    spin::Float64
    abs_spin::Float64
    natoms::Int
    symm::String
    flag1::String
    flag2::String
    flag3::String
end

_pfloat(x) = parse(Float64, x)
_pint(x) = parse(Int, x)

"""
    ShelxTITL(s::AbstractString)

Construct the TITL information from a String
"""
function ShelxTITL(s::AbstractString)
    tokens = split(strip(s))
    ShelxTITL(
        tokens[2],
        _pfloat(tokens[3]),
        _pfloat(tokens[4]),
        _pfloat(tokens[5]),
        _pfloat(tokens[6]),
        _pfloat(tokens[7]),
        _pint(tokens[8]),
        tokens[9],
        tokens[10],
        tokens[11],
        tokens[12],
    )
end


"""
    ShelxRecord

Representation for A SHELX record
"""
struct ShelxRecord <: AbstractRecord
    fname::String
    offset::Int
    length::Int
    titl::ShelxTITL
    comp::Composition
    reduced_comp::Composition
    function ShelxRecord(fname, offset, length, titl, comp)
        new(fname, offset, length, titl, comp, reduce_composition(comp))
    end
end

record_energy(s::ShelxRecord) = s.titl.enthalpy
record_comp(s::ShelxRecord) = s.comp
record_reduced_comp(s::ShelxRecord) = s.reduced_comp
record_id(s::ShelxRecord) = s.titl.label


"""
    read_shelx_record(fnames::AbstractVector)

Read all SHELX records from a list of (packed) files.
The SHELX record include the saved information of the structure with out
loading the full structure (positions of the atoms) into the memory.
"""
function read_shelx_record(fnames::AbstractVector)
    output = ShelxRecord[]
    for name in fnames
        tmp = open(name) do handle
            read_shelx_record(handle, name)
        end
        append!(output, tmp)
    end
    output
end

read_shelx_record(fname::AbstractString="*.res") = read_shelx_record(glob(fname))

"""
    read_shelx_record(io::IO, fname::AbstractString)

Read all SHELX records from a IO stream.
"""
function read_shelx_record(io::IO, fname::AbstractString)

    records = ShelxRecord[]
    local titl
    offset = 0
    symbols = Symbol[]
    capture = false
    last_pos = position(io)
    for line in eachline(io)
        if startswith(line, "TITL")
            titl = ShelxTITL(line)
            offset = last_pos
            last_pos = position(io)
            continue
        end
        if startswith(line, "SFAC")
            capture = true
            last_pos = position(io)
            continue
        end

        if startswith(line, "END")
            push!(
                records,
                ShelxRecord(
                    fname,
                    offset,
                    position(io) - offset,
                    titl,
                    Composition(countmap(symbols)),
                ),
            )
            empty!(symbols)
            capture = false
            last_pos = position(io)
            continue
        end

        if capture
            push!(symbols, Symbol(split(line)[1]))
        end
        last_pos = position(io)
    end
    records
end

"""
    extract_res(entries::Vector{ShelxRecord}, needle;outdir=".";outdir=".", save=true, outfile=nothing)

Write a SHELX entry to the disk from a haystack. 
Return the selected entries.

## Arguments:

- `needle`: `String` or `Regex` for selecting records based on their labels.
- `save`: If set to `true` (default), write the files out, otherwise just return the entries.
- `outdir`: Which output directory to use when writing out individual SHELX files.
- `outfile`: Name of the same file that will contain all the selected records. Default to `nothing` which 
  means that the output files will be named after the selected records. 

Note: 
This can results in undefined behaviour if non-identical records
share the same *label*.
"""
function extract_res(entries::Vector{ShelxRecord}, needle=""; outdir=".", save=true, outfile=nothing)
    if needle == ""
        selected = entries
    end
    selected = filter(x -> contains(x.titl.label, needle), entries)
    if !save
        return selected
    end

    # TODO - this is not good with lots of files
    # There can be a limit of open file handles on some systems.
    fnames = unique([x.fname for x in selected])
    ioset = Dict(name => open(name) for name in fnames)
    if outfile === nothing
        # Extract to individual files
        for entry in selected
            label = entry.titl.label
            outfile = joinpath(outdir, label * ".res")
            if isfile(outfile)
                @warn "Skipping existing file $(outfile)..."
                continue
            end
            open(outfile, "w") do fh
                stream = ioset[entry.fname]
                seek(stream, entry.offset)
                write(fh, read(stream, entry.length))
            end
        end
    else
        # Write to a single file
        open(outfile, "w") do fh
            for entry in selected
                stream = ioset[entry.fname]
                seek(stream, entry.offset)
                write(fh, read(stream, entry.length))
            end
        end
    end
    # Close the input file handles
    map(close, values(ioset))
    selected
end


"""
    CellBase.read_res(record::ShelxRecord)

Read the underlying record into a `Cell` object.
"""
function CellBase.read_res(record::ShelxRecord)
    open(record.fname) do fhandle
        seek(fhandle, record.offset)
        data = String(read(fhandle, record.length))
        read_res(split(data, "\n"))
    end
end


"""
    CellBase.read_res_many(records::Vector{ShelxRecord})

Read multiple records from single/multiple files.
"""
function CellBase.read_res_many(records::Vector{ShelxRecord})
    fnames_map = Dict()
    # Organise by the fnames
    for name in unique(x.fname for x in records)
        fnames_map[name] = findall(x -> x.fname == name, records)
    end
    @assert sum(x -> length(x), values(fnames_map)) == length(records)
    # Open the file handles
    out = Vector{Cell{Float64}}(undef, length(records)) 
    for (fname, vec_i) in fnames_map
        open(fname) do stream
            for i in vec_i 
                entry = records[i]
                seek(stream, entry.offset)
                data = String(read(stream, entry.length))
                out[i] = read_res(split(data, "\n"))
            end
        end
    end
    out
end

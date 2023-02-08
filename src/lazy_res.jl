#=
Code for fast loading SHELX dataset 

The idea is to not load the structure but only scan the text for the information
about the composition and the TITL line. 

Composition parsing takes a significant amount of time at the moment.
For even faster loading, the composition string should be encoded inside the REM information....

=#

using CellBase
using StatsBase

"""
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
Representation for A SHELX record
"""
struct ShelxRecord
    fname::String
    offset::Int
    length::Int
    titl::ShelxTITL
    comp::Composition
end


"""
    read_shelx_record(fnames)

Read all SHELX records from a list of (packed) files.
"""
function read_shelx_record(fnames)
    output = ShelxRecord[]
    for name in fnames
        tmp = open(name) do handle
            read_shelx_record(handle, name)
        end
        append!(output, tmp)
    end
    output
end

"""
    read_shelx_record(io::IO, fname::AbstractString)

Read all SHELX records from a IO stream.
"""
function read_shelx_record(io::IO, fname::AbstractString)

    records = Any[]
    local titl
    offset = 0
    symbols = Symbol[]
    capture = false

    for line in eachline(io)
        if startswith(line, "TITL")
            titl = ShelxTITL(line)
            offset = position(io)
            continue
        end
        if startswith(line, "SFAC")
            capture = true
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
            continue
        end

        if capture
            push!(symbols, Symbol(split(line)[1]))
        end
    end
    records
end

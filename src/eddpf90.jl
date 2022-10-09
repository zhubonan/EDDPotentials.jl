#=
Interface for the EDDP Fortran package
=# 

"""
Write the feature vectors
"""
function write_features(fc::FeatureContainer, outfile)
    ntot = length(fc)
    fsize = feature_size(fc.feature)
    pw = join(string.(fc.feature.two_body[1].p), " ") 
    rmax = suggest_rcut(fc.feature;offset=1.0)
    open(outfile, "w") do io
        for i in 1:ntot
            feat = fc.fvecs[i]
            metadata = fc.metadata[i]
            label = fc.labels[i]
            write_features(io, fsize, feat, metadata, label, rmax, pw)
        end
    end
end

function write_features(io::IO, fsize, fvec, metadata, label, rmax, pw)
    comp = metadata[:formula]
    pressure = metadata[:pressure]
    volume = metadata[:volume]
    enthalpy = metadata[:enthalpy]
    nc = size(fvec, 2)
    fl = "$(fsize[2]) $(fsize[3]) 0 0"
    metaline = "  structure: $(label)  composition: $(comp)  pressure: $(pressure)  volume: $(volume)  enthalpy: $(enthalpy)  rmax: $(rmax)  centers: $(nc)  length: $(fl)  powers: $(pw)\n"
    write(io, metaline)
    for fv in eachcol(fvec)
        # One body
        write(io, join(string.(fv[1:fsize[1]], " ")))
        write(io, "\n")
        # Two body
        write(io, join(string.(fv[fsize[1]+1:fsize[1] + fsize[2]], " ")))
        write(io, "\n")
        # Three body
        write(io, join(string.(fv[fsize[1]+1+fsize[2]:fsize[1] + fsize[2] + fsize[3]], " ")))
        write(io, "\n")
    end
end

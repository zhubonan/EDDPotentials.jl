#=
Interface for the EDDP Fortran package
=# 
using CellBase
using Printf

"""
Write the feature vectors
"""
function write_features(fc::FeatureContainer, outfile)
    ntot = length(fc)
    cf = fc.feature
    # flock's feature format is per-composition case
    # This maynot be correct here because p, q, rcut is allowed to vary per-composition  case... 
    nfeats = zeros(Int, 4)
    nfeats[1] = length(cf.elements)
    nfeats[2] = length(cf.two_body)
    nfeats[3] = length(cf.three_body)


    fsize_eddp = zeros(Int, 4)
    fsize_eddp[1] = nfeatures(cf.two_body[1])
    fsize_eddp[2] = nfeatures(cf.three_body[1])
    @assert nfeats[1] + nfeats[2] * fsize_eddp[1] + nfeats[3] * fsize_eddp[2] == nfeatures(cf)
    pw = join(string.(fc.feature.two_body[1].p), " ") 
    rmax = suggest_rcut(fc.feature;offset=1.0)
    open(outfile, "w") do io
        for i in 1:ntot
            feat = fc.fvecs[i]
            metadata = fc.metadata[i]
            label = fc.labels[i]
            H = fc.H[i]
            comps = CellBase.Composition(string(metadata[:formula]))
            comp = join(string.(comps.species), "-")
            write_features(io, nfeats, fsize_eddp, feat, label, rmax, pw; 
                           comp=comp, 
                           pressure=metadata[:pressure], 
                           volume=metadata[:volume], 
                           enthalpy=H)
        end
    end
end

function write_features(io::IO, nfeats, fsize_eddp, fvec, label, rmax, pw;comp, pressure, volume, enthalpy)
    nc = size(fvec, 2)
    fl = "$(fsize_eddp[1]) $(fsize_eddp[2]) 0 0"
    metaline = "  structure: $(label)  composition: $(comp)  pressure: $(pressure)  volume: $(volume)  enthalpy: $(enthalpy)  rmax: $(rmax)  centers: $(nc)  length: $(fl)  powers: $(pw)\n"
    write(io, metaline)
    for fv in eachcol(fvec)
        # One body
        #write(io, join(string.(fv[1:fsize[1]], " ")))
        for i in 1:nfeats[1]
            @printf io "%.14f " fv[i]
        end
        write(io, "\n")
        j = 1
        for _ in 1:nfeats[2]
            _i = nfeats[1] + j
            for i in _i:_i + fsize_eddp[1]-1
                @printf io "%.14f " fv[i]
            end
            j += fsize_eddp[1]
            write(io, "\n")
        end
        # Three bodies
        j = 1
        for _ in 1:nfeats[3]
            _i = nfeats[1] + j + fsize_eddp[1] * nfeats[2]
            for i in _i:_i + fsize_eddp[2]-1
                @printf io "%.14f " fv[i]
            end
            j += fsize_eddp[2]
            write(io, "\n")
        end
    end
end
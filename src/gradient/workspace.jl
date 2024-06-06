"""
    GradientWorkspace{T}

Buffer for storing forces and stress and support their calculations
"""
struct HardcoreWorkspace{T,N}
    "Hard core forces"
    fcore::Array{T,2}
    "Hard core energies"
    ecore::Array{T,1}
    "Hard core stresses"
    score::Array{T,3}
    core::N
end


function HardcoreWorkspace(T::DataType, core::N, nat::Int, ndims=3) where {N}
    core === nothing ? _nat = 0 : _nat = nat
    HardcoreWorkspace{T,N}(
        zeros(T, ndims, _nat),
        zeros(T, _nat),
        zeros(T, ndims, ndims, _nat),
        core,
    )
end

"""
    reset!(hc::HardcoreWorkspace)

    Reset all the arrays in the workspace to zero.
"""
function reset!(hc::HardcoreWorkspace)
    for prop in propertynames(hc)
        if isa(getproperty(hc, prop), AbstractArray)
            fill!(getproperty(hc, prop), 0)
        end
    end
    hc
end


"""
    GradientWorkspace{T,N}

Storage for gradients and other intermediate results.
Note that the intermediate gradients stored in gvec are in the shape of (ndims, nf, nn_max, nat).
Where nn_max is the maximum number of unique neighbour atoms.
The gradient stored are changes of the feature vector of atom i with respect to the moment of each
of its unique neighbours j and itself. 
This is NOT the same as the change of the feature vector j with respect to the moment of atom i.
"""
struct GradientWorkspace{T,N}
    "The array that stores the features"
    fvec::Matrix{T}
    "Temp array for dFi/drj' with the shape (ndims, nf, nn_max, nat)"
    gvec::Array{T,4}
    "Temp array to store the index of uunique neighbours for each atom"
    gvec_index::Array{Int,2}
    "Temp array to store the number of unique neighbours for each atom"
    gvec_nn::Array{Int,1}
    "Temp array for dF/dσ"
    stotv::Array{T,4}
    "Calculated forces"
    forces::Array{T,2}
    "Calculated stress"
    stress::Array{T,3}
    "Energy from features"
    energies::Array{T,1}
    "Calculated total forces"
    tot_forces::Array{T,2}
    "Calculated total stress"
    tot_stress::Array{T,2}
    "Per-atom energy"
    tot_energies::Array{T,1}
    hardcore::HardcoreWorkspace{T,N}
    do_grad::Bool
    one_body_offset::Int
end



"""
Initialise a workspace for computing forces

# Args
- `nf`: Number of features
- `nat`: Number of atoms in the unit cell
- `nn_max`: maximum number of unique neighbour atoms  default to nat + 1
- `ndims` (optional): The number of dimensions (3).
- `core` (optional): hard core potential.
"""
function GradientWorkspace(
    fvec::Matrix{T},
    nn_max=size(fvec, 2) + 1;
    ndims=3,
    core=nothing,
    do_grad=true,
    one_body_offset=0,
) where {T}
    nf, nat = size(fvec)
    _nat = nat
    if do_grad == false
        _nat = 0
    end
    GradientWorkspace(
        fvec,
        zeros(T, ndims, nf, nn_max, _nat), # gvec
        zeros(Int, nn_max, _nat), # neigh_index
        zeros(Int, _nat), # number of unique neighbours (include self)
        zeros(T, ndims, ndims, nf, _nat), # stotv
        zeros(T, ndims, _nat),  # forces
        zeros(T, ndims, ndims, _nat), # stress (per atom)
        zeros(T, nat), # Per atom energy from features
        zeros(T, ndims, _nat),  # total forces
        zeros(T, ndims, ndims), # total stress (global)
        zeros(T, _nat), # total energies per atom
        HardcoreWorkspace(T, core, nat, ndims),
        do_grad,
        one_body_offset,
    )
end

"""
    reset!(fb::GradientWorkspace)

Reset all the arrays in the workspace to zero.
"""
function reset!(fb::GradientWorkspace)
    reset!(fb.hardcore)
    for prop in propertynames(fb)
        if isa(getproperty(fb, prop), AbstractArray)
            # Avoid overwrite the one body part of the features
            if prop == :fvec
                getproperty(fb, prop)[fb.one_body_offset+1:end, :] .= 0
            else
                fill!(getproperty(fb, prop), 0)
            end
        end
    end
    fb
end


"""
    get_gvec_fj_ri(fb::GradientWorkspace)

    Recover the gradient in the dFj/dri format.
See `GradientWorkspace` for details on how the gradients are stored.
"""
function get_gvec_fj_ri(fb::GradientWorkspace)
    # Recover the gradient in the dFj/dri format
    gvec = fb.gvec
    gvec_fj_ri = zeros(size(gvec, 1), size(gvec, 2), size(gvec, 4), size(gvec, 4))
    for iat in axes(gvec, 4)
        for j = 1:fb.gvec_nn[iat]
            # translate neighbour local index to atom index
            jat = fb.gvec_index[j, iat]  # index of the atoms that has moved
            for a in axes(gvec, 1), b in axes(gvec, 2)
                gvec_fj_ri[a, b, iat, jat] += gvec[a, b, j, iat]
            end
        end
    end
    return gvec_fj_ri
end

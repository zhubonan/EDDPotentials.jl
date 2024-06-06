"""
    GradientWorkspace{T}

Buffer for storing forces and stress and support their calculations
"""
struct HardcoreWorkspace{T, N}
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
    HardcoreWorkspace{T, N}(
        zeros(T, ndims, _nat),
        zeros(T, _nat),
        zeros(T, ndims, ndims, _nat),
        core,
    )
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
    gvec_index::Array{Int, 2}
    "Temp array to store the number of unique neighbours for each atom"
    gvec_nn::Array{Int, 1}
    "Temp array for dF/dσ"
    stotv::Array{T,4}
    "Calculated forces"
    forces::Array{T,2}
    "Calculated stress"
    stress::Array{T,3}
    "Energy from features"
    energies::Array{T, 1}
    "Calculated total forces"
    tot_forces::Array{T,2}
    "Calculated total stress"
    tot_stress::Array{T,2}
    "Per-atom energy"
    tot_energies::Array{T, 1}
    hardcore::HardcoreWorkspace{T, N}
    do_grad::Bool
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
function GradientWorkspace(fvec::Matrix{T}, nn_max=nat + 1; 
    ndims=3, core=nothing, do_grad=true)  where {T}
    nf, nat= size(fvec)
    GradientWorkspace(
        fvec,
        zeros(T, ndims, nf, nn_max, nat), # gvec
        zeros(Int, nn_max, nat), # neigh_index
        zeros(Int, nat), # number of unique neighbours (include self)
        zeros(T, ndims, ndims, nf, nat), # stotv
        zeros(T, ndims, nat),  # forces
        zeros(T, ndims, ndims, nat), # stress (per atom)
        zeros(T, nat), # Per atom energy from features
        zeros(T, ndims, nat),  # total forces
        zeros(T, ndims, ndims), # total stress (global)
        zeros(T, nat), # total energies per atom
        HardcoreWorkspace(T, core, nat, ndims),
        do_grad,
    )
end


# TODO: Need updating
function clear!(fb::GradientWorkspace)
    fill!(fb.fvec, 0)
    fill!(fb.gvec, 0)
    fill!(fb.stotv, 0)
    fill!(fb.gvec_index, 0)
    fill!(fb.gvec_nn, 0)
    fb
end

# TODO: Need updating
function reset!(fb::GradientWorkspace)
    clear!(fb)
    for prop in [:fcore, :score, :ecore, :forces, :stress]
        fill!(getproperty(fb, prop), 0)
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
        for j in 1:fb.gvec_nn[iat]
            # translate neighbour local index to atom index
            jat = fb.gvec_index[j, iat]  # index of the atoms that has moved
            for a in axes(gvec, 1), b in axes(gvec, 2)
                gvec_fj_ri[a, b, iat, jat] += gvec[a, b, j, iat]
            end
        end
    end
    return gvec_fj_ri
end
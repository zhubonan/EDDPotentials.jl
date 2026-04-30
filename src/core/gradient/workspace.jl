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
    core_repulsion::N
end


function HardcoreWorkspace(T::DataType, core::N, nat::Int, ndims=3) where {N}
    core === nothing ? _nat = 0 : _nat = nat
    stress_dims = 3
    HardcoreWorkspace{T,N}(
        zeros(T, ndims, _nat),
        zeros(T, _nat),
        zeros(T, stress_dims, stress_dims, _nat),
        core,
    )
end

"""
    reset!(hc::HardcoreWorkspace)

    Reset all the arrays in the workspace to zero.
"""
function reset!(hc::HardcoreWorkspace)
    fill!(hc.fcore, 0)
    fill!(hc.score, 0)
    fill!(hc.ecore, 0)
    hc
end


"""
    FeatureBuffers{T}

Per-task buffers for feature computation in `compute_fv_gv_one!`.
Stored in a `Vector` indexed by chunk/thread slot for thread-safety.
Allocated lazily on first use.
"""
struct FeatureBuffers{T}
    "Cached pij values for three-body computation"
    pij::Matrix{T}
    "Cached pik values for three-body computation"
    pik::Matrix{T}
    "Cached qjk values for three-body computation"
    qjk::Matrix{T}
    "Cached inverse fij values"
    inv_fij::Vector{T}
    "Cached inverse fik values"
    inv_fik::Vector{T}
    "Cached inverse fjk values"
    inv_fjk::Vector{T}
    "Buffer for unique neighbour indices (per-atom deduplication)"
    neighbour_indices::Vector{Int}
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
    "Per-task feature computation buffers (lazily allocated, slot-indexed for thread-safety)"
    feat_bufs::Vector{FeatureBuffers{T}}
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
    stress_ndims = 3
    GradientWorkspace(
        fvec,
        zeros(T, ndims, nf, nn_max, _nat), # gvec
        zeros(Int, nn_max, _nat), # neigh_index
        zeros(Int, _nat), # number of unique neighbours (include self)
        zeros(T, stress_ndims, stress_ndims, nf, _nat), # stotv
        zeros(T, ndims, _nat),  # forces
        zeros(T, stress_ndims, stress_ndims, _nat), # stress (per atom)
        zeros(T, _nat), # Per atom energy from features
        zeros(T, ndims, _nat),  # total forces
        zeros(T, stress_ndims, stress_ndims), # total stress (global)
        zeros(T, _nat), # total energies per atom
        HardcoreWorkspace(T, core, nat, ndims),
        do_grad,
        one_body_offset,
        FeatureBuffers{T}[],  # feat_bufs — lazily allocated
    )
end

"""
    reset!(fb::GradientWorkspace)

Reset all the arrays in the workspace to zero.
"""
function reset!(fb::GradientWorkspace)
    fb.fvec[fb.one_body_offset+1:end, :] .= 0.0
    if fb.do_grad
        reset!(fb.hardcore)
        #fill!(fb.gvec, 0.)  # Reset on-demand inside compute_fv_gv_one!
        fill!(fb.gvec_index, 0)
        fill!(fb.gvec_nn, 0)
        #fill!(fb.stotv, 0.)  # Reset on-demand inside compute_fv_gv_one!
        fill!(fb.forces, 0.0)
        fill!(fb.stress, 0.0)
        fill!(fb.energies, 0.0)
        fill!(fb.tot_forces, 0.0)
        fill!(fb.tot_stress, 0.0)
        fill!(fb.tot_energies, 0.0)
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

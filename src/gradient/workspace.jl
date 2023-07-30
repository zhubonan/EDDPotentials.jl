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

struct GradientWorkspace{T,N}
    fvec::Vector{T}
    "Temp array for dF/dri for each neighbouring atoms"
    gvec::Array{T,3}
    "Temp array to store the index of neighbouring atoms"
    gvec_index::IndexVector
    "Temp array for dF/dσ"
    stotv::Array{T,3}
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
- `nn_max`: maximum number of unique neighbour atoms 
- `ndims` (optional): The number of dimensions (3).
- `core` (optional): hard core potential.
"""
function GradientWorkspace(fvec::Vector{T}, nat::Int, nn_max=min(nat, 100); 
    ndims=3, core=nothing, do_grad=true)  where {T}
    _fvec = similar(fvec)
    fill!(_fvec, 0)
    nf = size(fvec, 1)
    if do_grad
        GradientWorkspace(
            _fvec,
            zeros(T, ndims, nf, nn_max), # gvec
            IndexVector(nn_max), # neigh_index
            zeros(T, ndims, ndims, nf), # stotv
            zeros(T, ndims, nat),  # forces
            zeros(T, ndims, ndims, nat), # stress (per atom)
            zeros(T, nat), # Per atom energy from features
            zeros(T, ndims, nat),  # total forces
            zeros(T, ndims, ndims), # total stress (global)
            zeros(T, nat), # total energies per atom
            HardcoreWorkspace(T, core, nat, ndims),
            do_grad,
        )
    else
        GradientWorkspace(
            _fvec,
            zeros(T, ndims, nf, 0), # gvec
            IndexVector(0), # neigh_index
            zeros(T, ndims, ndims, 0), # stotv
            zeros(T, ndims, 0),  # forces
            zeros(T, ndims, ndims, 0), # stress (per atom)
            zeros(T, nat), # total energies per atom
            zeros(T, ndims, 0),  # total forces
            zeros(T, ndims, ndims), # total stress (global)
            zeros(T, nat), # per atom energies
            HardcoreWorkspace(T, core, nat, ndims),
            do_grad,
        )
    end
end

GradientWorkspace(fvec::Matrix, args...;kwargs...) = GradientWorkspace(fvec[:, 1], size(fvec, 2), args...;kwargs...)

function clear!(fb::GradientWorkspace)
    fill!(fb.fvec, 0)
    fill!(fb.gvec, 0)
    fill!(fb.stotv, 0)
    clear!(fb.gvec_index)
    fb
end

function reset!(fb::GradientWorkspace)
    clear!(fb)
    for prop in [:fcore, :score, :ecore, :forces, :stress]
        fill!(getproperty(fb, prop), 0)
    end
    fb
end
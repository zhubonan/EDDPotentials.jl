module EDDPTools

using Requires

function __init__()
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
        @info "Loading ASE bridge"
        include("ase_bridge.jl")
        @info "Loading Phonopy bridge"
        include("phonopy_bridge.jl")
    end
    @require Molly="aa0f7f06-fcc0-5ec4-a7f3-a573f33f9c4c" begin
        @info "Loading Molly-related stuff"
        include("molly.jl")
    end
end


end # module EDDPTools
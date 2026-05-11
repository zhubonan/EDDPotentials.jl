using JSON
using Flux: Chain, Dense, tanh, tanh_fast

"""
    ModelArtifact

Container returned by `load_model_artifact` for self-contained model archives.
"""
struct ModelArtifact
    model::AbstractNNInterface
    feature_spec::Union{Nothing,AbstractCellFeatureSpec}
    elemental_energies::Dict{Symbol,Float64}
    metadata::Dict{String,Any}
end

"""
    FeatureArtifact

Container returned by `load_feature_artifact`.
"""
struct FeatureArtifact
    feature_spec::AbstractCellFeatureSpec
    elemental_energies::Dict{Symbol,Float64}
end

const _MODEL_FORMAT = "eddp-model"
const _FEATURE_FORMAT = "eddp-feature"
const _FORMAT_VERSION = 2

_model_producer() = Dict(
    "package" => "EDDPotentials",
    "module" => "EDDPotentialsCore",
    "julia_version" => string(VERSION),
)

_require_key(data::AbstractDict, key::AbstractString) =
    get(data, key) do
        throw(ArgumentError("Missing required key `$key` in JSON artifact."))
    end

_basis_function_name(::typeof(fr)) = "fr"
_basis_function_name(::typeof(gfr)) = "gfr"
_basis_function_name(f) =
    throw(ArgumentError("Unsupported basis function for JSON serialization: $(repr(f))."))

function _basis_function_from_name(name::AbstractString)
    if name == "fr"
        return fr
    elseif name == "gfr"
        return gfr
    end
    throw(ArgumentError("Unsupported basis function name `$name` in JSON artifact."))
end

_activation_name(::typeof(identity)) = "identity"
_activation_name(::typeof(tanh)) = "tanh"
_activation_name(::typeof(tanh_fast)) = "tanh_fast"
_activation_name(::typeof(leaky_tanh)) = "leaky_tanh"
_activation_name(σ) =
    throw(ArgumentError("Unsupported activation for JSON serialization: $(repr(σ))."))

function _activation_from_name(name::AbstractString)
    if name == "identity"
        return identity
    elseif name == "tanh"
        return tanh
    elseif name == "tanh_fast"
        return tanh_fast
    elseif name == "leaky_tanh"
        return leaky_tanh
    end
    throw(ArgumentError("Unsupported activation name `$name` in JSON artifact."))
end

_array_to_data(x::AbstractVector) = collect(x)
_array_to_data(x::AbstractMatrix) = [collect(row) for row in eachrow(x)]
_array_to_data(x::Tuple) = collect(x)

function _dtype_name(T::Type)
    if T === Float64
        return "Float64"
    elseif T === Float32
        return "Float32"
    elseif T === Int
        return "Int"
    elseif T === Int64
        return "Int64"
    elseif T === Int32
        return "Int32"
    end
    throw(ArgumentError("Unsupported dtype for JSON serialization: $T."))
end

function _dtype_from_name(name::AbstractString)
    if name == "Float64"
        return Float64
    elseif name == "Float32"
        return Float32
    elseif name == "Int" || name == "Int64"
        return Int64
    elseif name == "Int32"
        return Int32
    end
    throw(ArgumentError("Unsupported dtype name `$name` in JSON artifact."))
end

function _typed_vector_to_data(x)
    T = isempty(x) ? Float64 : eltype(x)
    Dict(
        "dtype" => _dtype_name(T),
        "data" => _array_to_data(x),
    )
end

function _typed_matrix_to_data(x)
    Dict(
        "dtype" => _dtype_name(eltype(x)),
        "data" => _array_to_data(x),
    )
end

function _typed_vector_from_data(data; default_dtype=Float64)
    if data isa AbstractDict
        T = _dtype_from_name(_require_key(data, "dtype"))
        vals = _require_key(data, "data")
        # Handle case where empty arrays were serialized as {} (JSON.Object)
        if vals isa AbstractDict
            return T[]
        end
        out = Vector{T}(undef, length(vals))
        out .= T.(vals)
        return out
    end
    out = Vector{default_dtype}(undef, length(data))
    out .= default_dtype.(data)
    return out
end

function _matrix_from_rows(rows, ::Type{T}=Float64) where {T}
    nrow = length(rows)
    ncol = nrow == 0 ? 0 : length(rows[1])
    out = Matrix{T}(undef, nrow, ncol)
    for i = 1:nrow
        row = rows[i]
        # Handle case where empty arrays were serialized as {} (JSON.Object)
        if row isa AbstractDict
            out[i, :] .= T[]
        else
            length(row) == ncol ||
                throw(ArgumentError("Inconsistent row lengths in serialized matrix."))
            out[i, :] .= T.(row)
        end
    end
    out
end

function _typed_matrix_from_data(data; default_dtype=Float64)
    if data isa AbstractDict
        T = _dtype_from_name(_require_key(data, "dtype"))
        return _matrix_from_rows(_require_key(data, "data"), T)
    end
    return _matrix_from_rows(data, default_dtype)
end

function _json_compatible(x::Nothing)
    nothing
end

function _json_compatible(x::Union{Bool,String,AbstractString,Real})
    x
end

_json_compatible(x::Symbol) = String(x)
_json_compatible(x::AbstractVector) = [_json_compatible(v) for v in x]
_json_compatible(x::Tuple) = [_json_compatible(v) for v in x]
_json_compatible(x::AbstractMatrix) = _array_to_data(x)

function _json_compatible(x::AbstractDict)
    Dict(String(key) => _json_compatible(value) for (key, value) in pairs(x))
end

function _json_compatible(x::NamedTuple)
    Dict(String(key) => _json_compatible(value) for (key, value) in pairs(x))
end

function _json_compatible(x)
    throw(ArgumentError("Unsupported JSON metadata value of type $(typeof(x))."))
end

function _elemental_energies_to_data(energies::AbstractDict)
    Dict(String(key) => Float64(value) for (key, value) in pairs(energies))
end

function _elemental_energies_from_data(data)
    data === nothing && return Dict{Symbol,Float64}()
    Dict(Symbol(key) => Float64(value) for (key, value) in pairs(data))
end

function _feature_to_data(cf::CellFeature)
    d = Dict(
        "feature_type" => "cell_feature",
        "dtype" => begin
            if !isempty(cf.p2)
                _dtype_name(eltype(cf.p2))
            elseif !isempty(cf.p3)
                _dtype_name(eltype(cf.p3))
            else
                "Float64"
            end
        end,
        "elements" => string.(cf.elements),
        "p2" => _typed_vector_to_data(cf.p2),
        "rcut2" => cf.rcut2,
        "f2" => _basis_function_name(cf.f2),
        "g2" => _basis_function_name(cf.g2),
    )
    if !isempty(cf.p3)
        d["p3"] = _typed_vector_to_data(cf.p3)
        d["q3"] = _typed_vector_to_data(cf.q3)
        d["rcut3"] = cf.rcut3
        d["f3"] = _basis_function_name(cf.f3)
        d["g3"] = _basis_function_name(cf.g3)
    end
    d
end

function _feature_to_data(cf::RandomEmbeddedCellFeature)
    basis = cf.cell_feature
    d = Dict(
        "feature_type" => "random_embedded_cell_feature",
        "dtype" => _dtype_name(eltype(cf.embedding)),
        "elements" => string.(cf.elements),
        "nembed" => embedding_size(cf),
        "seed" => get(cf.constructor_kwargs, :seed, 0),
        "embedding" => _typed_matrix_to_data(cf.embedding),
        "p2" => _typed_vector_to_data(basis.p2),
        "rcut2" => basis.rcut2,
        "f2" => _basis_function_name(basis.f2),
        "g2" => _basis_function_name(basis.g2),
    )
    if !isempty(basis.p3)
        d["p3"] = _typed_vector_to_data(basis.p3)
        d["q3"] = _typed_vector_to_data(basis.q3)
        d["rcut3"] = basis.rcut3
        d["f3"] = _basis_function_name(basis.f3)
        d["g3"] = _basis_function_name(basis.g3)
    end
    d
end

function _feature_to_data(cf::MultiRandomEmbeddedCellFeature)
    Dict(
        "feature_type" => "multi_random_embedded_cell_feature",
        "embeddings" => [_feature_to_data(emb) for emb in cf.embeddings],
    )
end

function _feature_from_data(data)
    feature_type = _require_key(data, "feature_type")

    if feature_type == "multi_random_embedded_cell_feature"
        embeddings_data = _require_key(data, "embeddings")
        embeddings = [_feature_from_data(d) for d in embeddings_data]
        return MultiRandomEmbeddedCellFeature(embeddings)
    end

    elements = Symbol.(_require_key(data, "elements"))
    feature_dtype = haskey(data, "dtype") ? _dtype_from_name(data["dtype"]) : Float64

    if feature_type == "cell_feature"
        # New format: p2/rcut2/f2/g2 at top level
        p2 = _typed_vector_from_data(data["p2"])
        rcut2 = Float64(data["rcut2"])
        f2 = _basis_function_from_name(data["f2"])
        g2 = _basis_function_from_name(data["g2"])
        # Three-body: present if p3 key exists
        if haskey(data, "p3") && !isempty(data["p3"])
            p3 = _typed_vector_from_data(data["p3"])
            q3 = _typed_vector_from_data(data["q3"])
            rcut3 = Float64(data["rcut3"])
            f3 = _basis_function_from_name(data["f3"])
            g3 = _basis_function_from_name(data["g3"])
        else
            p3 = feature_dtype[]
            q3 = feature_dtype[]
            rcut3 = rcut2
            f3 = fr
            g3 = gfr
        end
        return CellFeature(
            elements;
            p2, p3, q3, rcut2, rcut3, f2, g2, f3, g3,
            geometry_sequence=false,
        )
    elseif feature_type == "random_embedded_cell_feature"
        nembed = Int(_require_key(data, "nembed"))
        seed = Int(get(data, "seed", 0))
        p2 = _typed_vector_from_data(_require_key(data, "p2"))
        rcut2 = Float64(_require_key(data, "rcut2"))
        f2 = _basis_function_from_name(_require_key(data, "f2"))
        g2 = _basis_function_from_name(_require_key(data, "g2"))
        if haskey(data, "p3") && !isempty(data["p3"])
            p3 = _typed_vector_from_data(data["p3"])
            q3 = _typed_vector_from_data(data["q3"])
            rcut3 = Float64(data["rcut3"])
            f3 = _basis_function_from_name(data["f3"])
            g3 = _basis_function_from_name(data["g3"])
        else
            p3 = feature_dtype[]
            q3 = feature_dtype[]
            rcut3 = rcut2
            f3 = fr
            g3 = gfr
        end
        cf = RandomEmbeddedCellFeature(
            elements;
            nembed,
            seed,
            T=feature_dtype,
            p2,
            p3,
            q3,
            rcut2,
            rcut3,
            f2,
            g2,
            f3,
            g3,
            geometry_sequence=false,
            optimize_embedding=false,
        )
        cf.embedding .= feature_dtype.(_typed_matrix_from_data(_require_key(data, "embedding")))
        return cf
    end

    throw(ArgumentError("Unsupported feature_type `$feature_type` in JSON artifact."))
end

function _transform_to_data(x::Nothing)
    nothing
end

function _transform_to_data(x::ZScoreTransform)
    Dict(
        "type" => "zscore",
        "len" => x.len,
        "dims" => x.dims,
        "mean" => _typed_vector_to_data(x.mean),
        "scale" => _typed_vector_to_data(x.scale),
    )
end

function _transform_from_data(data)
    data === nothing && return nothing
    _require_key(data, "type") == "zscore" ||
        throw(ArgumentError("Unsupported transform type `$(_require_key(data, "type"))`."))
    mean_vec = _typed_vector_from_data(_require_key(data, "mean"))
    scale_vec = _typed_vector_from_data(_require_key(data, "scale"))
    ZScoreTransform(
        Int(_require_key(data, "len")),
        Int(get(data, "dims", 2)),
        mean_vec,
        scale_vec,
    )
end

function _layer_to_data(layer::Dense)
    Dict(
        "layer_type" => "dense",
        "activation" => _activation_name(layer.σ),
        "weight" => _typed_matrix_to_data(layer.weight),
        "bias" => _typed_vector_to_data(layer.bias),
    )
end

function _layer_to_data(layer::CellEmbedding)
    Dict(
        "layer_type" => "cell_embedding",
        "dtype" => _dtype_name(eltype(layer.two_body.weight)),
        "n" => layer.n,
        "m" => layer.m,
        "two_body_weight" => _typed_matrix_to_data(layer.two_body.weight),
        "three_body_weight" => _typed_matrix_to_data(layer.three_body.weight),
    )
end

function _layer_to_data(layer)
    throw(ArgumentError("Unsupported layer type for JSON serialization: $(typeof(layer))."))
end

function _layer_from_data(data, feature_spec::AbstractCellFeatureSpec)
    layer_type = _require_key(data, "layer_type")
    if layer_type == "dense"
        weight = _typed_matrix_from_data(_require_key(data, "weight"))
        bias = _typed_vector_from_data(_require_key(data, "bias"); default_dtype=eltype(weight))
        σ = _activation_from_name(_require_key(data, "activation"))
        return Dense(weight, bias, σ)
    elseif layer_type == "cell_embedding"
        feature_spec isa CellFeature || throw(
            ArgumentError("CellEmbedding JSON layer requires a CellFeature feature spec."),
        )
        two_body_weight = _typed_matrix_from_data(_require_key(data, "two_body_weight"))
        three_body_weight = _typed_matrix_from_data(_require_key(data, "three_body_weight"))
        T = haskey(data, "dtype") ? _dtype_from_name(data["dtype"]) : eltype(two_body_weight)
        layer = CellEmbedding(
            T,
            feature_spec,
            Int(_require_key(data, "n")),
            Int(_require_key(data, "m")),
        )
        layer.two_body.weight .= T.(two_body_weight)
        layer.three_body.weight .= T.(three_body_weight)
        return layer
    end
    throw(ArgumentError("Unsupported layer_type `$layer_type` in JSON artifact."))
end

function _model_to_data(
    model::LinearInterface;
    feature_spec::AbstractCellFeatureSpec,
    elemental_energies::Dict{Symbol,Float64},
    metadata::Dict{String,Any},
)
    Dict(
        "interface_type" => "linear",
        "feature_spec" => _feature_to_data(feature_spec),
        "elemental_energies" => _elemental_energies_to_data(elemental_energies),
        "metadata" => _json_compatible(metadata),
        "param" => _typed_matrix_to_data(model.param),
    )
end

function _model_to_data(
    model::ManualFluxBackPropInterface;
    feature_spec::AbstractCellFeatureSpec,
    elemental_energies::Dict{Symbol,Float64},
    metadata::Dict{String,Any},
)
    Dict(
        "interface_type" => "manual_flux_backprop",
        "feature_spec" => _feature_to_data(feature_spec),
        "elemental_energies" => _elemental_energies_to_data(elemental_energies),
        "metadata" => _json_compatible(metadata),
        "xt" => _transform_to_data(model.xt),
        "yt" => _transform_to_data(model.yt),
        "layers" => [_layer_to_data(layer) for layer in model.chain.layers],
    )
end

function _model_to_data(
    model::EnsembleNNInterface;
    feature_spec::AbstractCellFeatureSpec,
    elemental_energies::Dict{Symbol,Float64},
    metadata::Dict{String,Any},
)
    Dict(
        "interface_type" => "ensemble",
        "feature_spec" => _feature_to_data(feature_spec),
        "elemental_energies" => _elemental_energies_to_data(elemental_energies),
        "metadata" => _json_compatible(metadata),
        "weights" => _array_to_data(model.weights),
        "models" => [_submodel_to_data(submodel) for submodel in model.models],
    )
end

function _model_to_data(
    model::AbstractNNInterface;
    feature_spec::AbstractCellFeatureSpec,
    elemental_energies::Dict{Symbol,Float64},
    metadata::Dict{String,Any},
)
    throw(ArgumentError("Unsupported model type for JSON serialization: $(typeof(model))."))
end

_submodel_to_data(model::LinearInterface) =
    Dict("interface_type" => "linear", "param" => _typed_matrix_to_data(model.param))

function _submodel_to_data(model::ManualFluxBackPropInterface)
    Dict(
        "interface_type" => "manual_flux_backprop",
        "xt" => _transform_to_data(model.xt),
        "yt" => _transform_to_data(model.yt),
        "layers" => [_layer_to_data(layer) for layer in model.chain.layers],
    )
end

function _submodel_to_data(model::AbstractNNInterface)
    throw(ArgumentError("Unsupported ensemble submodel type for JSON serialization: $(typeof(model))."))
end

function _submodel_from_data(data, feature_spec)
    interface_type = _require_key(data, "interface_type")
    if interface_type == "linear"
        return LinearInterface(_typed_matrix_from_data(_require_key(data, "param")))
    elseif interface_type == "manual_flux_backprop"
        layers = [_layer_from_data(layer, feature_spec) for layer in _require_key(data, "layers")]
        chain = Chain(layers...)
        return ManualFluxBackPropInterface(
            chain;
            xt=_transform_from_data(get(data, "xt", nothing)),
            yt=_transform_from_data(get(data, "yt", nothing)),
        )
    end
    throw(ArgumentError("Unsupported ensemble submodel interface_type `$interface_type`."))
end

function _model_from_data(data)
    interface_type = _require_key(data, "interface_type")
    feature_spec = _feature_from_data(_require_key(data, "feature_spec"))
    elemental_energies = _elemental_energies_from_data(
        get(data, "elemental_energies", Dict{String,Float64}()),
    )
    metadata = Dict{String,Any}(get(data, "metadata", Dict{String,Any}()))

    if interface_type == "linear"
        model = LinearInterface(_typed_matrix_from_data(_require_key(data, "param")))
    elseif interface_type == "manual_flux_backprop"
        layers = [_layer_from_data(layer, feature_spec) for layer in _require_key(data, "layers")]
        chain = Chain(layers...)
        model = ManualFluxBackPropInterface(
            chain;
            xt=_transform_from_data(get(data, "xt", nothing)),
            yt=_transform_from_data(get(data, "yt", nothing)),
        )
    elseif interface_type == "ensemble"
        models = [
            _submodel_from_data(model_data, feature_spec) for
            model_data in _require_key(data, "models")
        ]
        weights = Float64.(_require_key(data, "weights"))
        length(models) == length(weights) ||
            throw(ArgumentError("Serialized ensemble has mismatched model and weight counts."))
        model = EnsembleNNInterface(Tuple(models), weights)
    else
        throw(ArgumentError("Unsupported interface_type `$interface_type` in JSON artifact."))
    end

    ModelArtifact(model, feature_spec, elemental_energies, metadata)
end

function save_model_artifact(
    path::AbstractString,
    model::AbstractNNInterface;
    feature_spec=nothing,
    elemental_energies=Dict{Symbol,Float64}(),
    metadata=nothing,
)
    feature_spec isa AbstractCellFeatureSpec || throw(
        ArgumentError("`feature_spec` must be provided when saving a JSON model artifact."),
    )
    md = metadata === nothing ? Dict{String,Any}() : Dict{String,Any}(_json_compatible(metadata))
    energies = Dict{Symbol,Float64}(Symbol(key) => Float64(value) for (key, value) in pairs(elemental_energies))
    data = Dict(
        "format" => _MODEL_FORMAT,
        "format_version" => _FORMAT_VERSION,
        "producer" => _model_producer(),
        "model" => _model_to_data(
            model;
            feature_spec,
            elemental_energies=energies,
            metadata=md,
        ),
    )
    open(path, "w") do io
        JSON.print(io, data, 4)
    end
    path
end

function load_model(path::AbstractString)
    load_model_artifact(path).model
end

function load_model_artifact(path::AbstractString)
    if endswith(path, ".json")
        data = JSON.parsefile(path)
        _require_key(data, "format") == _MODEL_FORMAT ||
            throw(ArgumentError("File `$path` is not an EDDP model JSON artifact."))
        Int(_require_key(data, "format_version")) <= _FORMAT_VERSION ||
            throw(ArgumentError("Unsupported EDDP model JSON format version in `$path`."))
        return _model_from_data(_require_key(data, "model"))
    elseif endswith(path, ".jld2")
        return jldopen(path) do f
            metadata = Dict{String,Any}()
            for key in ("builder_uuid", "train_labels", "val_labels", "test_labels")
                if key in keys(f)
                    metadata[key] = f[key]
                end
            end
            feature_spec = "cf" in keys(f) ? f["cf"] : nothing
            elemental_energies = "elemental_energies" in keys(f) ?
                                 _elemental_energies_from_data(f["elemental_energies"]) :
                                 Dict{Symbol,Float64}()
            ModelArtifact(load_from_jld2(f), feature_spec, elemental_energies, metadata)
        end
    end
    throw(ArgumentError("Unsupported model artifact path `$path`. Expected .json or .jld2."))
end

function save_feature_json(
    path::AbstractString,
    feature_spec::AbstractCellFeatureSpec;
    elemental_energies=Dict{Symbol,Float64}(),
)
    data = Dict(
        "format" => _FEATURE_FORMAT,
        "format_version" => _FORMAT_VERSION,
        "producer" => _model_producer(),
        "feature" => _feature_to_data(feature_spec),
        "elemental_energies" => _elemental_energies_to_data(elemental_energies),
    )
    open(path, "w") do io
        JSON.print(io, data, 4)
    end
    path
end

function load_feature_artifact(path::AbstractString)
    data = JSON.parsefile(path)
    _require_key(data, "format") == _FEATURE_FORMAT ||
        throw(ArgumentError("File `$path` is not an EDDP feature JSON artifact."))
    Int(_require_key(data, "format_version")) == _FORMAT_VERSION ||
        throw(ArgumentError("Unsupported EDDP feature JSON format version in `$path`."))
    FeatureArtifact(
        _feature_from_data(_require_key(data, "feature")),
        _elemental_energies_from_data(get(data, "elemental_energies", Dict{String,Float64}())),
    )
end

function load_feature_json(path::AbstractString)
    load_feature_artifact(path).feature_spec
end

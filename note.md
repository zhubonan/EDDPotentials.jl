
Pattern for using `ForwardDiff.jl` for jacobian calculation.
Can probably write some kind of macro for code generation.....
But hand written is sufficient fow small scale networks.

For larger networks, we can always use backprop + SDG....


```julia
# Manually compute the gradients
model_net = Chain(Dense(16=>8, tanh;bias=true), Dense(8=>1))
function f!(o, param)
    pack_param!(model_net, param)
    o .= mean.(model_net.(x_train_norm)) .- y_train_norm
    o
end

predictf(x) = mean.(model_net.(x))

"Unpack parameter"
function unpack_param(model)
    n = 0
    for layer in model.layers
        n += length(layer.weight) + length(layer.bias)
    end
    out = zeros(eltype(model.layers[1].weight), n)
    i = 1
    for layer in model.layers
        l = length(layer.weight)
        out[i: i+l-1] .= vec(layer.weight)
        i += l
        l = length(layer.bias)
        out[i: i+l-1] .= vec(layer.bias)
        i +=l
    end
    out
end

"Update the parameters of the model"
function pack_param!(model, param)
    i = 1
    for layer in model.layers
        l = length(layer.weight)
        layer.weight[:] .= param[i: i+l-1]
        i += l
        l = length(layer.bias)
        layer.bias[:] .= param[i: i+l-1]
        i +=l
    end
    model
end

function net_manual(param)
    pairs = []
    i = 1
    for layer in model_net
        for key in (:weight, :bias)
            elm = getproperty(layer, key)
            n = length(elm)
            push!(pairs, i=>i+n-1)
            i += n
        end
    end
    w1 = param[pairs[1].first:pairs[1].second]
    b1 = param[pairs[2].first:pairs[2].second]
    w2 = param[pairs[3].first:pairs[3].second]
    b2 = param[pairs[4].first:pairs[4].second]

    f(data) = w2  * model_net[1].σ(w1 * data .+ b1) .+ b2 |> model_net[2].σ
    mean.(f.(x_train_norm))
end

function ftmp_w1(x)
    layer1 = model_net.layers[1]
    layer2 = model_net.layers[2]
    f2(data) = (x * data) .+ layer1.bias .|> layer1.σ |> layer2
    mean.(f2.( x_train_norm))
end

function ftmp_b1(x)
    layer1 = model_net.layers[1]
    layer2 = model_net.layers[2]
    f2(data) = (layer1.weight * data) .+ x .|> layer1.σ |> layer2
    mean.(f2.( x_train_norm))
end



function ftmp_w2(x)
    layer1 = model_net.layers[1]
    layer2 = model_net.layers[2]
    f2(data) = x * layer1(data)  .+ layer2.bias |> layer2.σ 
    mean.(f2.( x_train_norm))
end

function ftmp_b2(x)
    layer1 = model_net.layers[1]
    layer2 = model_net.layers[2]
    f2(data) =  layer2.weight * layer1(data) .+ x |> layer2.σ 
    mean.(f2.( x_train_norm))
end

#ftmp_all(model_net.layers[1].weight)
#hcat(ForwardDiff.jacobian(ftmp_w1, model_net.layers[1].weight),
#ForwardDiff.jacobian(ftmp_b1, model_net.layers[1].bias),
#ForwardDiff.jacobian(ftmp_w2, model_net.layers[2].weight),
#ForwardDiff.jacobian(ftmp_b2, model_net.layers[2].bias),
#    )


function model_many_g!(g, params)
    # apply the parameters
    pack_param!(model_net, params)
    
    ifunc = 1
    funs = [ftmp_w1, ftmp_b1, ftmp_w2, ftmp_b2]
    i = 1
    # Construct the jacobian
    for layer in model_net
        for key in (:weight, :bias)
            elm = getproperty(layer, key)
            n = length(elm)
            g[:, i:i+n-1] .= ForwardDiff.jacobian(funs[ifunc], elm)
            ifunc += 1
            i += n
        end
    end
    g
end
```


### Code Snippet - Improve `ForwardDiff` based approach

The key is to build a `Chain(Dense...)` with the correct type of the elements on demand.
`ForwardDiff` works by replacing the array with `Dual` type. This means that we cannot update an existing
`Dense` with different type, but instead the type needs to be inferred from the input.

An alternative is to pre-built the net with the `Dual` type, however, `Dual` contains a `Tag` as the type parameter
so this might not work?

Most of the time on the minimisation is spent on the jacobian evaluation....

```julia
model_net = Chain(Dense(16=>8, tanh;bias=true), Dense(8=>1))

"Unpack a flat vector into the shape of a given matrix"
function unpack_matrix(param, mat, offset=1)
    l = length(mat)
    reshape(param[offset:offset+l - 1], size(mat))
end

function diffwt(wt)
    net = Chain(Dense(wt), 
                Dense(size(wt, 1)=> 1))
        
    mean(net(x_train_norm[1]))
end

"""
Allow forward diff to be used
"""
function diffparam(param::Vector{T};x_train_norm=x_train_norm) where {T}
    i = 1
    nets = []
    for layer in model_net
        wt::Matrix{T} = unpack_matrix(param, layer.weight, i)
        i += length(wt)
        bias::Vector{T} = unpack_matrix(param, layer.bias, i)
        i += length(bias)
        layer_ = Dense(wt, bias, layer.σ)
        push!(nets, layer_)
    end

    f = Chain(nets...)
    
    total::Matrix{T} = reduce(hcat, x_train_norm)
    all_E::Matrix{T} = f(total)
    
    out::Vector{T} = zeros(T, length(x_train_norm))
    ct = 1
    for i in 1:length(out)
        lv = size(x_train_norm[i], 2)
        out[i] = mean(all_E[ct:ct+lv-1])
        ct += lv
    end
    out
end

targetf(x) = diffparam(x; x_train_norm)
cfg1 = JacobianConfig(t, x, Chunk{145}());

function g!(g, x)
    ForwardDiff.jacobian!(g, t, x, cfg1)
end
#ForwardDiff.jacobian(diffparam, rand(145)) 

"Update the parameters of the model"
function pack_param!(model, param)
    i = 1
    for layer in model.layers
        l = length(layer.weight)
        layer.weight[:] .= param[i: i+l-1]
        i += l
        l = length(layer.bias)
        layer.bias[:] .= param[i: i+l-1]
        i +=l
    end
    model
end

"Unpack parameter"
function unpack_param(model)
    n = 0
    for layer in model.layers
        n += length(layer.weight) + length(layer.bias)
    end
    out = zeros(eltype(model.layers[1].weight), n)
    i = 1
    for layer in model.layers
        l = length(layer.weight)
        out[i: i+l-1] .= vec(layer.weight)
        i += l
        l = length(layer.bias)
        out[i: i+l-1] .= vec(layer.bias)
        i +=l
    end
    out
end


function f!(o, param)
    pack_param!(model_net, param)
    o .= mean.(model_net.(x_train_norm)) .- y_train_norm
    o
end

p0 = unpack_param(model_net)

od = OnceDifferentiable(f!, g!, p0, f!(zeros(Float32, 9000), p0); inplace=true)
```
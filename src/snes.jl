#=
Implementation of the SNES algorithm
=#  
using Random
using Distributions


function snes(f, vec, population_size, max_iteration;
    σ=ones(size(vec)...), η_μ=1., 
    η_σ=(3+log(length(vec)))/(5 * sqrt(length(vec))),
    verbose=false,
    u_k_function=_u_k,
    callback=nothing
    ) 

    n = length(vec)
    s_k_list = zeros(n, population_size)
    det_μ_J =  zeros(n)
    det_σ_J =  zeros(n)
    iter = 1
    while iter <= max_iteration
        # Compute the fitness 
        fitness = map(1:population_size) do i
            randn!(@view s_k_list[:, i])
            z_k = vec .+ σ .* s_k_list[:, i]
            f(z_k)
        end

        # Compute the utility function values with the best one being ranked 1st
        rank = sortperm(sortperm(fitness))
        u_k = u_k_function(rank)

        # update
        fill!(det_μ_J, 0)
        fill!(det_σ_J, 0)
        for i in axes(s_k_list, 2)  # Each candidate
            det_μ_J .+= s_k_list[:, i] .* u_k[i]
            det_σ_J .+= (s_k_list[:, i] .^ 2 .- 1) .* u_k[i]
        end

        # Update
        vec .+= η_μ .* σ .* det_μ_J
        σ .*= exp.(η_σ/2 .* det_σ_J)
        verbose && @info "Iteration $iter: Fitness is $(f(vec))"
#        verbose && @info "Iteration $iter: σ is $σ"
#        verbose && @info "Iteration $iter: Fitness $(minimum(fitness))"
        isnothing(callback) || callback()
        iter += 1
    end
end


"""
The utility function
"""
function _u_k(indices)
    n = length(indices)
    deno = sum(indices) do i
        max(0, log(n / 2 +1) - log(i))
    end
    map(indices) do i
        max(0, log(n /2 + 1) - log(i)) / deno
    end
end
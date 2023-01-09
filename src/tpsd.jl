import Optim

struct TwoPointSteepestDescent{F} <: Optim.FirstOrderOptimizer
    α_init::F
    manifold::Optim.Manifold
end

TwoPointSteepestDescent(; α_init=1e-8, manifold=Optim.Flat()) =
    TwoPointSteepestDescent(α_init, manifold)

mutable struct TPSDState <: Optim.AbstractOptimizerState
    x::Any
    x_previous::Any
    g_x_previous::Any
    f_x_previous::Any
    is_first_cycle::Any
end

function Optim.initial_state(method::TwoPointSteepestDescent, options, d, initial_x)
    x0 = similar(initial_x)
    fill!(x0, 0)
    gx = similar(initial_x)
    fill!(gx, 0)
    TPSDState(copy(initial_x), x0, gx, value(d), true)
end

function Optim.update_state!(
    d,
    state::TPSDState,
    method::TwoPointSteepestDescent{T},
) where {T}
    # Compute gradient and value
    value_gradient!!(d, state.x)

    gx = Optim.gradient(d)
    # changes k vs k-1
    dx = state.x .- state.x_previous
    dg = gx - state.g_x_previous
    if !state.is_first_cycle
        α = abs(dot(dx, dg) / (dot(dg, dg) + floatmin(eltype(state.x))))
    else
        α = method.α_init
        state.is_first_cycle = false
    end

    # Update variables
    copy!(state.x_previous, state.x)
    copy!(state.g_x_previous, gx)
    state.f_x_previous = value(d)

    # Take step (k+1)
    state.x .= state.x .- α .* gx

    false
end

function Optim.trace!(
    tr,
    d,
    state,
    iteration,
    method::TwoPointSteepestDescent,
    options,
    curr_time=time(),
)
    Optim.common_trace!(tr, d, state, iteration, method, options, curr_time)
end

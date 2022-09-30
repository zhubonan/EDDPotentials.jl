#=
Interface for Neutron network implementations
=#
using StatsBase: ZScoreTransform, fit, transform!, reconstruct!

abstract type AbstractNNInterface end

"Return a vector of the gradients against the parameters."
function gradparam! end

"Return the gradient against the inputs."
function gradinp! end

"Do the forward step"
function forward! end

"Do the backward step"
function backward! end

"Return a vector containing the parameters"
function paramvector end

"Return a vector containing the parameters"
function paramvector! end

"Return the number of parameters"
function nparams end

"Set the parameters with a vector"
function setparamvector! end

## Standardisation

include("manual_backprop.jl")
include("linear.jl")
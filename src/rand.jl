#= 
Generation of random stuff
=#

using Random

const _RNG = MersenneTwister()

"Reseed the global random generator"
reseed!(seed) = Random.seed!(_RNG, seed)
reseed!() = Random.seed!(_RNG)

"Return a random float 3 vector"
rand3f() = rand(_RNG, 3)
"Return a random float 3 vector within the range" 
rand3f(x, y) = rand(_RNG, 3) .* (y - x) .+ x
"Return a random float between 0 and 1"
randf() = rand(_RNG, Float64)
"Return a random float with range (x, y)"
randf(x, y) = x  + (y - x) * rand(_RNG, Float64)
"Fill vector x with random permulation indices"
randp!(x) = randperm!(_RNG, x)
"Alias for randf(x, y)"
rand_range(x, y) = randf(x, y)

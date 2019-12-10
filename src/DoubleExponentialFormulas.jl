module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    quadde,
    quadts,
    quades,
    quadss,
    QuadTS,
    QuadES,
    QuadSS

const h0inv = 8

function generate_tables end
function trapez end

function quadde end

include("quadts.jl")
function quadts end

include("quades.jl")
function quades end

include("quadss.jl")
function quadss end


"""
    startindex(f::Function, weights, istart::Integer)

Returns the first index i which `f(weights[i][1])` is not NaN,
where `i >= istart`.

This function is employed to avoid sampling `f(x)` with too large `abs(x)` in
trapezoidal rule.
"""
function startindex(f::Function, weights, istart::Integer)
    iend = length(weights)
    for i in istart:iend
        x, _ = @inbounds weights[i]
        if all(isnotnan.(f(x)))
            return i
        end
    end
    return length(weights)
end

isnotnan(x) = !isnan(x)


end # module

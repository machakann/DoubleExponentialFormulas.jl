module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    QuadTS,
    QuadES,
    QuadSS,
    QuadDE,
    quadde


"""
    startindex(f::Function, weights, istart::Integer)

Returns the first index i which `f(weights[i][1])` is not NaN,
where `i >= istart`.

This function is employed to avoid sampling `f(x)` with too large `abs(x)` in
trapezoidal rule. For example, the limit of `x*exp(-x^2)` may result in NaN
when `x` approaches infinity since the expression reduced to be `Inf*0` in
float point number computations.
"""
function startindex(f::Function, weights, istart::Integer)
    iend = length(weights)
    for i in istart:iend
        x, _ = @inbounds weights[i]
        y = try
            f(x)
        catch
            continue
        end
        all(.!isnan.(y)) && return i
    end
    return iend
end


"""
    sum_pairwise(f::Function, itr, istart::Integer=1, iend::Integer=length(itr))

Return total summation of items in `itr` from `istart`-th through `iend`-th
 with applying a function `f`. This function uses pairwise summation algorithm
 to reduce numerical error as possible.

NOTE: This function doesn't check `istart` and `iend`. Be careful to use.
"""
sum_pairwise(f::Function, itr, istart::Integer=1, iend::Integer=length(itr)) =
    Base.mapreduce_impl(f, +, itr, istart, iend)


"""
    estimate_error(T::Type{<:AbstractFloat}, prevI, I)

Estimate an error from the true integral(, though it may not be very accurate).

With double exponential formula, the error from the true value `I` decays
exponentially.
    ΔI(h) = I - I(h) ≈ exp(-C/h)
`I(h)` is the numerical integral with a step size `h`, `C` is a
constant depending on the integrand. If the step size is halved,
    ΔI(h/2) = I - I(h/2) ≈ exp(-2C/h) ≈ ( ΔI(h) )²,
the significant digits gets almost twice. Therefore, `I(h/2)` should be
considerably closer to the true value than `I(h)`.
    ΔI(h/2) ≈ ( ΔI(h) )² ≈ ( I - I(h) )² ≈ ( I(h/2) - I(h) )²
Introduce a magnification coefficient `M` for safety.
    ΔI(h/2) ≈ M*( I(h/2) - I(h) )²
Furthermore, the floating point numbers generally have a finite
significant digits; `ΔI(h/2)` would asymptotically get closer to `I*ε`
at minimum, rather than 0.
    ΔI(h/2) ≈ M*( I(h/2) - I(h) )² + I*ε
            ≈ M*( I(h/2) - I(h) )² + I(h/2)*ε

FIXME: The logical ground of the safety factor is very weak. Any better way?
"""
function estimate_error(T::Type{<:AbstractFloat}, prevI, I)
    ε = eps(T)
    M = 20
    return M*norm(I - prevI)^2 + norm(I)*ε
end


include("quadts.jl")
include("quades.jl")
include("quadss.jl")
include("quadde.jl")


end # module

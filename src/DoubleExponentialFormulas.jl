module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    QuadTS,
    QuadES,
    QuadSS,
    QuadDE,
    quadde


"""
    search_edge_t(T::Type{<:AbstractFloat}, ϕ::Function, ϕ′::Function)

Return the largest `t` satisfying
  ϕ(t) != 0 && ϕ(t) != Inf && ϕ′(t) != 0 && ϕ′(t) != Inf.
"""
function search_edge_t(T::Type{<:AbstractFloat}, ϕ::Function, ϕ′::Function)
    isover(t) = iszero(ϕ(t)) || isinf(ϕ(t)) || iszero(ϕ′(t)) || isinf(ϕ′(t))
    t = zero(T)
    dt0 = T(8)
    for _ in 1:1000
        isover(t + dt0) && break
        t += dt0
    end
    for i in 1:1000
        dt = dt0/2^i
        isover(t + dt) && continue
        t += dt
    end
    return t
end


function generate_table(samplepoint, maxlevel::Integer, n0::Int, tmax::T) where {T<:AbstractFloat}
    h0 = tmax/n0
    table0 = [samplepoint(k*h0) for k in 1:n0]
    reverse!(table0)
    tables = Vector{Tuple{T,T}}[]
    n = n0
    for _ in 1:maxlevel
        n *= 2
        h = tmax/n
        table = [samplepoint(k*h) for k in 1:2:n]
        reverse!(table)
        push!(tables, table)
    end
    return table0, tables
end


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


include("mapsum.jl")


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

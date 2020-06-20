using LinearAlgebra: norm
using Printf: @printf


"""
    QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)

A callable object to integrate a function over the range (-∞, ∞) using the
*sinh-sinh quadrature*. It utilizes the change of variables to transform the
integrand into a form well-suited to the trapezoidal rule.

`QuadSS` tries to calculate integral values `maxlevel` times at a maximum;
the step size of a trapezoid is started from `h0` and is halved in each
following repetition for finer accuracy. The repetition is terminated when the
difference from the previous estimation gets smaller than a certain threshold.
The threshold is determined by the runtime parameters, see below.

The type `T` represents the accuracy of interval. The integrand should accept
values `x<:T` as its parameter.

---

    I, E = (q::QuadSS)(f::Function;
                       atol::Real=zero(T),
                       rtol::Real=atol>0 ? zero(T) : sqrt(eps(T)))
                       where {T<:AbstractFloat}

Numerically integrate `f(x)` over the interval (-∞, ∞) and return the integral
value `I` and an estimated error `E`. The `E` is not exactly equal to the
difference from the true value. However, one can expect that the integral value
`I` is converged if `E <= max(atol, rtol*norm(I))` is true. Otherwise, the
obtained `I` would be unreliable; the number of repetitions exceeds the
`maxlevel` before converged.

The integrand `f` can also return any value other than a scalar, as far as
`+`, `-`, multiplication by real values, and `LinearAlgebra.norm`, are
implemented. For example, `Vector` or `Array` of numbers are acceptable
although, unfortunately, it may not be very performant.

# Examples
```jldoctest
julia> using DoubleExponentialFormulas

julia> using LinearAlgebra: norm

julia> qss = QuadSS(Float64);

julia> f(x) = 2/(1 + x^2);

julia> I, E = qss(f);

julia> I ≈ 2π
true

julia> E ≤ sqrt(eps(Float64))*norm(I)
true

julia> I, E = qss(x -> [1/(1 + x^2), 2/(1 + x^2)]);

julia> I ≈ [π, 2π]
true

julia> E ≤ sqrt(eps(Float64))*norm(I)
true
```
"""
struct QuadSS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    table0::Vector{Tuple{T,T}}
    tables::NTuple{N,Vector{Tuple{T,T}}}
end
function QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=12, h0::Real=one(T))
    @assert maxlevel > 0
    @assert h0 > 0
    origin = weight(QuadSS, zero(T))
    table0 = generate_table(QuadSS, h0, 1)
    tables = Vector{Tuple{T,T}}[]
    for level in 1:maxlevel
        h = h0/2^level
        table = generate_table(QuadSS, h, 2)
        push!(tables, table)
    end
    return QuadSS{T,maxlevel}(T(h0), origin, table0, Tuple(tables))
end

function (q::QuadSS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    f⁺ = f
    f⁻ = u -> f(-u)
    sample⁺(t) = f⁺(t[1])*t[2]
    sample⁻(t) = f⁻(t[1])*t[2]
    x0, w0 = q.origin
    Σ = f(x0)*w0
    istart⁺ = startindex(f⁺, q.table0, 1)
    istart⁻ = startindex(f⁻, q.table0, 1)
    Σ += mapsum(sample⁺, q.table0, istart⁺)
    Σ += mapsum(sample⁻, q.table0, istart⁻)
    h0 = q.h0
    I = h0*Σ
    E = zero(eltype(I))
    for level in 1:N
        table = q.tables[level]
        istart⁺ = startindex(f⁺, table, 2*istart⁺ - 1)
        istart⁻ = startindex(f⁻, table, 2*istart⁻ - 1)
        Σ += mapsum(sample⁺, table, istart⁺)
        Σ += mapsum(sample⁻, table, istart⁻)
        h = h0/2^level
        prevI = I
        I = h*Σ
        E = estimate_error(T, prevI, I)
        tol = max(norm(I)*rtol, atol)
        !(E > tol) && break
    end
    return I, E
end

function Base.show(io::IO, ::MIME"text/plain", q::QuadSS{T,N}) where {T<:AbstractFloat,N}
    @printf(io, "DoubleExponentialFormulas.QuadSS{%s}: maxlevel=%d, h0=%.3e",
            string(T), N, q.h0)
end


function generate_table(::Type{QuadSS}, h::T, step::Int) where {T<:AbstractFloat}
    table = Tuple{T,T}[]
    k = 1
    while true
        t = k*h
        xk, wk = weight(QuadSS, t)
        xk ≥ floatmax(T) && break
        wk ≥ floatmax(T) && break
        push!(table, (xk, wk))
        k += step
    end
    reverse!(table)
    return table
end


function weight(::Type{QuadSS}, t)
    ϕ(t) = sinh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*cosh(sinh(t)*π/2)
    return ϕ(t), ϕ′(t)
end

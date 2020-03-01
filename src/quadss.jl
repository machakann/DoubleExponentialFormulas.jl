using LinearAlgebra: norm
using Printf: @printf


"""
    QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)

A callable object to integrate a function over the range (-∞, ∞) using the
*sinh-sinh quadrature*. It utilizes the change of variables to transform the
integrand into a form well-suited to the trapezoidal rule.

`QuadSS` tries to calculate integral values `maxlevel` times at a maximum;
applying the trapezoidal rule with the initial division number `n0` and the
number is doubled in each following repetition for finer accuracy. The
repetition is terminated when the difference from the previous estimation gets
smaller than a certain threshold. The threshold is determined by the runtime
parameters, see below.

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
    n0::Int
    tmax::T
    origin::Tuple{T,T}
    table0::Vector{Tuple{T,T}}
    tables::NTuple{N,Vector{Tuple{T,T}}}
end
function QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=12, n0::Real=7)
    @assert maxlevel > 0
    @assert n0 > 0
    ϕ(t) = sinh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*cosh(sinh(t)*π/2)
    samplepoint(t) = (ϕ(t), ϕ′(t))
    tmax = search_edge_t(T, ϕ, ϕ′)
    origin = samplepoint(zero(T))
    table0, tables = generate_table(samplepoint, maxlevel, n0, tmax)
    return QuadSS{T,maxlevel}(n0, tmax, origin, table0, Tuple(tables))
end

function (q::QuadSS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    f⁺ = f
    f⁻ = u -> f(-u)
    sample⁺(t) = f⁺(t[1])*t[2]
    sample⁻(t) = f⁻(t[1])*t[2]
    x0, w0 = q.origin
    I = f(x0)*w0
    istart⁺ = startindex(f⁺, q.table0, 1)
    istart⁻ = startindex(f⁻, q.table0, 1)
    I += mapsum(sample⁺, q.table0, istart⁺)
    I += mapsum(sample⁻, q.table0, istart⁻)
    n0 = q.n0
    tmax = q.tmax
    h0 = tmax/n0
    Ih = I*h0
    E = zero(eltype(Ih))
    n = n0
    for level in 1:N
        table = q.tables[level]
        istart⁺ = startindex(f⁺, table, 2*istart⁺ - 1)
        istart⁻ = startindex(f⁻, table, 2*istart⁻ - 1)
        I += mapsum(sample⁺, table, istart⁺)
        I += mapsum(sample⁻, table, istart⁻)
        prevIh = Ih
        n *= 2
        h = tmax/n
        Ih = I*h
        E = estimate_error(T, prevIh, Ih)
        !(E > max(norm(Ih)*rtol, atol)) && break
    end
    return Ih, E
end

function Base.show(io::IO, ::MIME"text/plain", q::QuadSS{T,N}) where {T<:AbstractFloat,N}
    @printf("DoubleExponentialFormulas.QuadSS{%s}: maxlevel=%d, h0=%.3e",
            string(T), N, q.h0)
end

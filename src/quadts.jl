using LinearAlgebra: norm
using Printf: @printf


"""
    QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, n0::Integer=3)

A callable object to integrate a function over the range [-1, 1] using the
*tanh-sinh quadrature*. It utilizes the change of variables to transform the
integrand into a form well-suited to the trapezoidal rule.

`QuadTS` tries to calculate integral values `maxlevel` times at a maximum;
applying the trapezoidal rule with the initial division number `n0` and the
number is doubled in each following repetition for finer accuracy. The
repetition is terminated when the difference from the previous estimation gets
smaller than a certain threshold. The threshold is determined by the runtime
parameters, see below.

The type `T` represents the accuracy of interval. The integrand should accept
values `x<:T` as its parameter.

---

    I, E = (q::QuadTS)(f::Function;
                       atol::Real=zero(T),
                       rtol::Real=atol>0 ? zero(T) : sqrt(eps(T)))
                       where {T<:AbstractFloat}

Numerically integrate `f(x)` over the interval [-1, 1] and return the integral
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

julia> qts = QuadTS(Float64);

julia> f(x) = 2/(1 + x^2);

julia> I, E = qts(f);

julia> I ≈ π
true

julia> E ≤ sqrt(eps(Float64))*norm(I)
true

julia> I, E = qts(x -> [1/(1 + x^2), 2/(1 + x^2)]);

julia> I ≈ [π/2, π]
true

julia> E ≤ sqrt(eps(Float64))*norm(I)
true
```
"""
struct QuadTS{T<:AbstractFloat,N}
    n0::Int
    tmax::T
    origin::Tuple{T,T}
    table0::Vector{Tuple{T,T}}
    tables::NTuple{N,Vector{Tuple{T,T}}}
end
function QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=12, n0::Integer=3)
    @assert maxlevel > 0
    @assert n0 > 0
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    samplepoint(t) = (ϕ(t), ϕ′(t))
    tmax = asinh(atanh(1 - 2*eps(one(T)))*2/π)
    origin = samplepoint(zero(T))
    table0, tables = generate_table(samplepoint, maxlevel, n0, tmax)
    return QuadTS{T,maxlevel}(n0, tmax, origin, table0, Tuple(tables))
end

function (q::QuadTS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    sample(t) = f(t[1])*t[2] + f(-t[1])*t[2]
    x0, w0 = q.origin
    I = f(x0)*w0 + mapsum(sample, q.table0)
    n0 = q.n0
    tmax = q.tmax
    h0 = tmax/n0
    Ih = I*h0
    E = zero(eltype(Ih))
    n = n0
    for table in q.tables
        I += mapsum(sample, table)
        prevIh = Ih
        n *= 2
        h = tmax/n
        Ih = I*h
        E = estimate_error(T, prevIh, Ih)
        !(E > max(norm(Ih)*rtol, atol)) && break
    end
    return Ih, E
end


function Base.show(io::IO, ::MIME"text/plain", q::QuadTS{T,N}) where {T<:AbstractFloat,N}
    @printf("DoubleExponentialFormulas.QuadTS{%s}: maxlevel=%d, n0=%d",
            string(T), N, q.n0)
end

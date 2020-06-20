using LinearAlgebra: norm
using Printf: @printf


"""
    QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)

A callable object to integrate a function over the range [-1, 1] using the
*tanh-sinh quadrature*. It utilizes the change of variables to transform the
integrand into a form well-suited to the trapezoidal rule.

`QuadTS` tries to calculate integral values `maxlevel` times at a maximum;
the step size of a trapezoid is started from `h0` and is halved in each
following repetition for finer accuracy. The repetition is terminated when the
difference from the previous estimation gets smaller than a certain threshold.
The threshold is determined by the runtime parameters, see below.

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
    h0::T
    origin::Tuple{T,T}
    table0::Vector{Tuple{T,T}}
    tables::NTuple{N,Vector{Tuple{T,T}}}
end
function QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=12, h0::Real=one(T))
    @assert maxlevel > 0
    @assert h0 > 0
    origin = weight(QuadTS, zero(T))
    table0 = generate_table(QuadTS, h0, 1)
    tables = Vector{Tuple{T,T}}[]
    for level in 1:maxlevel
        h = h0/2^level
        table = generate_table(QuadTS, h, 2)
        push!(tables, table)
    end
    return QuadTS{T,maxlevel}(T(h0), origin, table0, Tuple(tables))
end

function (q::QuadTS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    sample(t) = f(t[1])*t[2] + f(-t[1])*t[2]
    x0, w0 = q.origin
    Σ = f(x0)*w0 + mapsum(sample, q.table0)
    h0 = q.h0
    I = h0*Σ
    E = zero(eltype(I))
    for level in 1:N
        table = q.tables[level]
        Σ += mapsum(sample, table)
        h = h0/2^level
        prevI = I
        I = h*Σ
        E = estimate_error(T, prevI, I)
        tol = max(norm(I)*rtol, atol)
        !(E > tol) && break
    end
    return I, E
end

function Base.show(io::IO, ::MIME"text/plain", q::QuadTS{T,N}) where {T<:AbstractFloat,N}
    @printf(io, "DoubleExponentialFormulas.QuadTS{%s}: maxlevel=%d, h0=%.3e",
            string(T), N, q.h0)
end


function generate_table(::Type{QuadTS}, h::T, step::Int) where {T<:AbstractFloat}
    table = Tuple{T,T}[]
    k = 1
    while true
        t = k*h
        xk, wk = weight(QuadTS, t)
        1 - xk ≤ eps(T) && break
        wk ≤ floatmin(T) && break
        push!(table, (xk, wk))
        k += step
    end
    reverse!(table)
    return table
end


function weight(::Type{QuadTS}, t)
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    return ϕ(t), ϕ′(t)
end

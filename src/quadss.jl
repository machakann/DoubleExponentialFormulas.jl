using LinearAlgebra: norm
using Printf: @printf


"The table for the sinh-sinh quadrature"
struct QuadSSWeightTable{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    table::Vector{Tuple{T,T}}
end
Base.size(wt::QuadSSWeightTable) = size(wt.table)
Base.getindex(wt::QuadSSWeightTable, i::Int) = getindex(wt.table, i)


"""
    QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)

A callable object to integrate a function over the range [-∞, ∞] using the
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

Numerically integrate `f(x)` over the interval [-∞, ∞] and return the integral
value `I` and an estimated error `E`. The `E` is not exactly equal to the
difference from the true value. However, one can expect that the integral value
`I` is converged if `E <= max(atol, rtol*norm(I))` is true. Otherwise, the
obtained `I` would be unreliable; the number of repetitions exceeds the
`maxlevel` before converged.

The integrand `f` can also return any value other than a scalar, as far as
[`+`](@ref), [`-`](@ref), multiplication by real values, and [`norm`](@ref),
are implemented. For example, `Vector` or `Array` of numbers are acceptable
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

julia> E ≤ sqrt(eps(I))*norm(I)
true

julia> g(x) = [1/(1 + x^2), 2/(1 + x^2)];

julia> I, E = qss(g);

julia> I ≈ [π, 2π]
true

julia> E ≤ sqrt(eps(eltype(I)))*norm(I)
true
```
"""
struct QuadSS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    tables::NTuple{N,QuadSSWeightTable{T}}
end
function QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)
    @assert maxlevel > 0
    t0 = zero(T)
    tables, origin = generate_tables(QuadSSWeightTable, maxlevel, T(h0))
    return QuadSS{T,maxlevel}(T(h0), origin, tables)
end

function (q::QuadSS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    f⁺ = f
    f⁻ = u -> f(-u)
    x0, w0 = q.origin
    I = f(x0)*w0
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    istart⁺ = 1
    istart⁻ = 1
    for level in 1:N
        table = q.tables[level]
        istart⁺ = startindex(f⁺, table, istart⁺)
        istart⁻ = startindex(f⁻, table, istart⁻)
        I += sum_pairwise(t -> f⁺(t[1])*t[2], table, istart⁺)
        I += sum_pairwise(t -> f⁻(t[1])*t[2], table, istart⁻)
        h = h0/2^(level - 1)
        prevIh = Ih
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 1 && break
        istart⁺ = 2*istart⁺ - 1
        istart⁻ = 2*istart⁻ - 1
    end
    return Ih, E
end

function Base.show(io::IO, ::MIME"text/plain", q::QuadSS{T,N}) where {T<:AbstractFloat,N}
    @printf("DoubleExponentialFormulas.QuadSS{%s}: maxlevel=%d, h0=%.3e",
            string(T), N, q.h0)
end


function generate_tables(::Type{QuadSSWeightTable}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = sinh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*cosh(sinh(t)*π/2)
    tables = Vector{QuadSSWeightTable}(undef, maxlevel)
    for level in 1:maxlevel
        table = Tuple{T,T}[]
        h = h0/2^(level - 1)
        k = 1
        step = level == 1 ? 1 : 2
        while true
            t = k*h
            xk = ϕ(t)
            xk ≥ floatmax(T) && break
            wk = ϕ′(t)
            wk ≥ floatmax(T) && break
            push!(table, (xk, wk))
            k += step
        end
        reverse!(table)
        tables[level] = QuadSSWeightTable{T}(table)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    return Tuple(tables), origin
end

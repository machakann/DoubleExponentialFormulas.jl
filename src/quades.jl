using LinearAlgebra: norm
using Printf: @printf


struct QuadESWeightTable{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    table::Vector{Tuple{T,T}}
end
Base.size(wt::QuadESWeightTable) = size(wt.table)
Base.getindex(wt::QuadESWeightTable, i::Int) = getindex(wt.table, i)


struct QuadES{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    tables⁺::NTuple{N,QuadESWeightTable{T}}
    tables⁻::NTuple{N,QuadESWeightTable{T}}
end
function QuadES(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)
    @assert maxlevel > 0
    t0 = zero(T)
    tables⁺, tables⁻, origin = generate_tables(QuadESWeightTable, maxlevel, T(h0))
    return QuadES{T,maxlevel}(T(h0), origin, tables⁺, tables⁻)
end

function (q::QuadES{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    x0, w0 = q.origin
    I = f(x0)*w0
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    istart⁺ = 1
    for level in 1:N
        table⁺ = q.tables⁺[level]
        table⁻ = q.tables⁻[level]
        istart⁺ = startindex(f, table⁺, istart⁺)
        I += sum_pairwise(t -> f(t[1])*t[2], table⁺, istart⁺)
        I += sum_pairwise(t -> f(t[1])*t[2], table⁻)
        h = h0/2^(level - 1)
        prevIh = Ih
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 1 && break
        istart⁺ = 2*istart⁺ - 1
    end
    return Ih, E
end

function Base.show(io::IO, ::MIME"text/plain", q::QuadES{T,N}) where {T<:AbstractFloat,N}
    @printf("DoubleExponentialFormulas.QuadES{%s}: maxlevel=%d, h0=%.3e",
            string(T), N, q.h0)
end


function generate_tables(::Type{QuadESWeightTable}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = exp(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*exp(sinh(t)*π/2)
    tables⁺ = Vector{QuadESWeightTable}(undef, maxlevel)
    tables⁻ = Vector{QuadESWeightTable}(undef, maxlevel)
    for level in 1:maxlevel
        h = h0/2^(level - 1)
        k = 1
        step = level == 1 ? 1 : 2
        table⁺ = Tuple{T,T}[]
        while true
            t = k*h
            xk = ϕ(t)
            xk ≥ floatmax(T) && break
            wk = ϕ′(t)
            wk ≥ floatmax(T) && break
            push!(table⁺, (xk, wk))
            k += step
        end

        k = -1
        step = level == 1 ? -1 : -2
        table⁻ = Tuple{T,T}[]
        while true
            t = k*h
            xk = ϕ(t)
            xk ≤ eps(T) && break
            wk = ϕ′(t)
            wk ≤ floatmin(T) && break
            push!(table⁻, (xk, wk))
            k += step
        end

        reverse!(table⁺)
        reverse!(table⁻)
        tables⁺[level] = QuadESWeightTable{T}(table⁺)
        tables⁻[level] = QuadESWeightTable{T}(table⁻)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    return Tuple(tables⁺), Tuple(tables⁻), origin
end

using LinearAlgebra: norm


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
function QuadES(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    tables⁺, tables⁻, origin = generate_tables(QuadESWeightTable, maxlevel, T(h0))
    QuadES{T,maxlevel+1}(T(h0), origin, tables⁺, tables⁻)
end

function (q::QuadES{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    x0, w0 = q.origin
    I = f(x0)*w0
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    sample(t) = f(t[1])*t[2]
    istart⁺ = 1
    for level in 0:(N-1)
        prevIh = Ih
        istart⁺ = startindex(f, q.tables⁺[1], istart⁺)
        iend⁺ = length(q.tables⁺[level+1])
        I += sum(sample, view(q.tables⁺[level+1], istart⁺:iend⁺))
        I += sum(sample, q.tables⁻[level+1])
        h = h0/2^level
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 0 && break
        istart⁺ = 2*istart⁺ - 1
    end
    Ih, E
end


function generate_tables(::Type{QuadESWeightTable}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = exp(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*exp(sinh(t)*π/2)
    tables⁺ = Vector{QuadESWeightTable}(undef, maxlevel+1)
    tables⁻ = Vector{QuadESWeightTable}(undef, maxlevel+1)
    for level in 0:maxlevel
        h = h0/2^level

        k = 1
        step = level == 0 ? 1 : 2
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
        step = level == 0 ? -1 : -2
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
        tables⁺[level+1] = QuadESWeightTable{T}(table⁺)
        tables⁻[level+1] = QuadESWeightTable{T}(table⁻)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(tables⁺), Tuple(tables⁻), origin
end

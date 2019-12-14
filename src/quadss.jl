using LinearAlgebra: norm


struct QuadSSWeightTable{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    table::Vector{Tuple{T,T}}
end
Base.size(wt::QuadSSWeightTable) = size(wt.table)
Base.getindex(wt::QuadSSWeightTable, i::Int) = getindex(wt.table, i)


struct QuadSS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    tables::NTuple{N,QuadSSWeightTable{T}}
end
function QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/8)
    @assert maxlevel > 0
    t0 = zero(T)
    tables, origin = generate_tables(QuadSSWeightTable, maxlevel, T(h0))
    QuadSS{T,maxlevel}(T(h0), origin, tables)
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
    Ih, E
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
    Tuple(tables), origin
end

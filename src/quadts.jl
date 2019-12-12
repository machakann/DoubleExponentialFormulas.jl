using LinearAlgebra: norm


struct QuadTSWeightTable{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    table::Vector{Tuple{T,T}}
end
Base.size(wt::QuadTSWeightTable) = size(wt.table)
Base.getindex(wt::QuadTSWeightTable, i::Int) = getindex(wt.table, i)


struct QuadTS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    tables::NTuple{N,QuadTSWeightTable{T}}
end
function QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    tables, origin = generate_tables(QuadTSWeightTable, maxlevel, T(h0))
    QuadTS{T,maxlevel+1}(T(h0), origin, tables)
end

function (q::QuadTS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    x0, w0 = q.origin
    I = f(x0)*w0
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    sample(t) = f(t[1])*t[2] + f(-t[1])*t[2]
    for level in 0:(N-1)
        prevIh = Ih
        I += sum(sample, q.tables[level+1])
        h = h0/2^level
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 0 && break
    end
    Ih, E
end
function (q::QuadTS{T,N})(f::Function, a::Real, b::Real;
                          atol=zero(T), kwargs...) where {T<:AbstractFloat,N}
    if a > b
        I, E = q(f, b, a; kwargs...)
        -I, E
    else
        if a == -1 && b == 1
            q(f; kwargs...)
        else
            a′ = T(a)
            b′ = T(b)
            s = b′ + a′
            t = b′ - a′
            atol′ = atol/t*2
            f′(u) = f((s + t*u)/2)
            I, E = q(f′; atol=atol′, kwargs...)
            I*t/2, E*t/2
        end
    end
end
function (q::QuadTS{T,N})(f::Function, a::Real, b::Real, c::Real...;
                          kwargs...) where {T<:AbstractFloat,N}
    I, E = q(f, a, b; kwargs...)
    bc = (b, c...)
    for i in 2:length(bc)
        dI, dE = q(f, bc[i-1], bc[i]; kwargs...)
        I += dI
        E += dE
    end
    I, E
end


function generate_tables(::Type{QuadTSWeightTable}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    tables = Vector{QuadTSWeightTable}(undef, maxlevel+1)
    for level in 0:maxlevel
        table = Tuple{T,T}[]
        h = h0/2^level
        k = 1
        step = level == 0 ? 1 : 2
        while true
            t = k*h
            xk = ϕ(t)
            1 - xk ≤ eps(T) && break
            wk = ϕ′(t)
            wk ≤ floatmin(T) && break
            push!(table, (xk, wk))
            k += step
        end
        reverse!(table)
        tables[level+1] = QuadTSWeightTable{T}(table)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(tables), origin
end

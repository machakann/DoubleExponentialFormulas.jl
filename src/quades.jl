using LinearAlgebra: norm


struct QuadESWeights{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    weights::Vector{Tuple{T,T}}
end
Base.size(qesw::QuadESWeights) = size(qesw.weights)
Base.getindex(qesw::QuadESWeights, i::Int) = getindex(qesw.weights, i)


struct QuadES{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    table_p::NTuple{N,QuadESWeights{T}}
    table_n::NTuple{N,QuadESWeights{T}}
end
function QuadES(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    table_p, table_n, origin = generate_table(QuadESWeights, maxlevel, T(h0))
    QuadES{T,maxlevel+1}(T(h0), origin, table_p, table_n)
end

function (q::QuadES{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    h0 = q.h0
    x0, w0 = q.origin
    I = f(x0)*w0
    istart_p = startindex(f, q.table_p[1], 1)
    I += trapez(f, q.table_p[1], I, istart_p)
    I += trapez(f, q.table_n[1], I, 1)
    Ih = I*h0
    E = zero(eltype(Ih))
    istart_p = max(1, istart_p - 1)
    for level in 1:(N-1)
        prevIh = Ih
        h = h0/2^level
        I += trapez(f, q.table_p[level+1], I, istart_p)
        I += trapez(f, q.table_n[level+1], I, 1)
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && break
        istart_p *= 2
    end
    Ih, E
end


function trapez(f::Function, qesw::QuadESWeights{T}, I,
                istart::Integer) where {T<:AbstractFloat}
    dI = zero(I)
    iend = length(qesw)
    for i in istart:iend
        x, w = qesw[i]
        dI += f(x)*w
    end
    dI
end


function generate_table(::Type{QuadESWeights}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = exp(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*exp(sinh(t)*π/2)
    table_p = Vector{QuadESWeights}(undef, maxlevel+1)
    table_n = Vector{QuadESWeights}(undef, maxlevel+1)
    for level in 0:maxlevel
        h = h0/2^level

        k = 1
        step = level == 0 ? 1 : 2
        weights_p = Tuple{T,T}[]
        while true
            t = k*h
            xk = ϕ(t)
            xk ≥ floatmax(T) && break
            wk = ϕ′(t)
            wk ≥ floatmax(T) && break
            push!(weights_p, (xk, wk))
            k += step
        end

        k = -1
        step = level == 0 ? -1 : -2
        weights_n = Tuple{T,T}[]
        while true
            t = k*h
            xk = ϕ(t)
            xk ≤ eps(T) && break
            wk = ϕ′(t)
            wk ≤ floatmin(T) && break
            push!(weights_n, (xk, wk))
            k += step
        end

        reverse!(weights_p)
        reverse!(weights_n)
        table_p[level+1] = QuadESWeights{T}(weights_p)
        table_n[level+1] = QuadESWeights{T}(weights_n)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(table_p), Tuple(table_n), origin
end

using LinearAlgebra: norm


struct QuadSSWeights{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    weights::Vector{Tuple{T,T}}
end
Base.size(qssw::QuadSSWeights) = size(qssw.weights)
Base.getindex(qssw::QuadSSWeights, i::Int) = getindex(qssw.weights, i)


struct QuadSS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    table::NTuple{N,QuadSSWeights{T}}
end
function QuadSS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    table, origin = generate_table(QuadSSWeights, maxlevel, T(h0))
    QuadSS{T,maxlevel+1}(T(h0), origin, table)
end

function (q::QuadSS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    f_p = f
    f_n = x -> f(-x)
    h0 = q.h0
    x0, w0 = q.origin
    I = f(x0)*w0
    istart_p = startindex(f_p, q.table[1], 1)
    istart_n = startindex(f_n, q.table[1], 1)
    I += trapez(f, q.table[1], I, istart_p)
    I += trapez(f, q.table[1], I, istart_n)
    Ih = I*h0
    E = zero(eltype(Ih))
    istart_p = max(1, istart_p - 1)
    istart_n = max(1, istart_n - 1)
    for level in 1:(N-1)
        prevIh = Ih
        h = h0/2^level
        istart_p = startindex(f_p, q.table[level+1], istart_p)
        istart_n = startindex(f_n, q.table[level+1], istart_n)
        I += trapez(f, q.table[level+1], I, istart_p)
        I += trapez(f, q.table[level+1], I, istart_n)
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && break
        istart_p *= 2
        istart_n *= 2
    end
    Ih, E
end


function trapez(f::Function, qssw::QuadSSWeights{T}, I,
                istart::Integer) where {T<:AbstractFloat}
    dI = zero(I)
    iend = length(qssw)
    for i in istart:iend
        x, w = qssw[i]
        dI += f(x)*w
    end
    dI
end


function generate_table(::Type{QuadSSWeights}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = sinh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*cosh(sinh(t)*π/2)
    table = Vector{QuadSSWeights}(undef, maxlevel+1)
    for level in 0:maxlevel
        weights = Tuple{T,T}[]
        h = h0/2^level
        k = 1
        step = level == 0 ? 1 : 2
        while true
            t = k*h
            xk = ϕ(t)
            xk ≥ floatmax(T) && break
            wk = ϕ′(t)
            wk ≥ floatmax(T) && break
            push!(weights, (xk, wk))
            k += step
        end
        reverse!(weights)
        table[level+1] = QuadSSWeights{T}(weights)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(table), origin
end

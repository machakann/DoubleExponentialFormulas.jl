using LinearAlgebra: norm


struct QuadSSWeights{T<:AbstractFloat}
    weights::Vector{Tuple{T,T}}
end


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
    h0 = q.h0
    x0, w0 = q.origin
    I0 = f(x0)*w0
    I, Ih = trapez(f, q.table[1], I0, h0, rtol, atol)
    E = zero(eltype(Ih))
    for level in 1:(N-1)
        prevIh = Ih
        h = h0/2^level
        I, Ih, tol = trapez(f, q.table[level+1], I, h, rtol, atol)
        E = norm(prevIh - Ih)
        !(E > tol) && break
    end
    Ih, E
end


function trapez(f::Function, qssw::QuadSSWeights{T}, I, h::T,
                rtol::Real, atol::Real) where {T<:AbstractFloat}
    tol = zero(float(eltype(I)))
    Ih = zero(I)
    for (x, w) in qssw.weights
        dI1 = f(x)*w
        dI2 = f(-x)*w
        I += dI1
        I += dI2
        Ih = I*h
        tol = max(norm(Ih)*rtol, atol)
        !(norm(dI1*h) + norm(dI2*h) > tol) && break
    end
    I, Ih, tol
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
        table[level+1] = QuadSSWeights{T}(weights)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(table), origin
end

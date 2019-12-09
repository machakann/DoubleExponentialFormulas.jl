using LinearAlgebra: norm


struct QuadESWeights{T<:AbstractFloat}
    weights_n::Vector{Tuple{T,T}}
    weights_p::Vector{Tuple{T,T}}
end


struct QuadES{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    table::NTuple{N,QuadESWeights{T}}
end
function QuadES(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    table, origin = generate_table(QuadESWeights, maxlevel, T(h0))
    QuadES{T,maxlevel+1}(T(h0), origin, table)
end

function (q::QuadES{T,N})(f::Function; atol::Real=zero(T),
                            rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    h0 = q.h0
    x0, w0 = q.origin
    I0 = f(x0)*w0
    I = trapez(f, q.table[1], I0)
    Ih = I*h0
    E = zero(eltype(Ih))
    for level in 1:(N-1)
        prevIh = Ih
        h = h0/2^level
        I = trapez(f, q.table[level+1], I)
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && break
    end
    Ih, E
end


function trapez(f::Function, qesw::QuadESWeights{T}, I) where {T<:AbstractFloat}
    dI = zero(I)
    for (x, w) in qesw.weights_n
        dI += f(x)*w
    end
    for (x, w) in qesw.weights_p
        dI += f(x)*w
    end
    I + dI
end


function generate_table(::Type{QuadESWeights}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = exp(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)*exp(sinh(t)*π/2)
    table = Vector{QuadESWeights}(undef, maxlevel+1)
    for level in 0:maxlevel
        h = h0/2^level

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

        reverse!(weights_n)
        reverse!(weights_p)
        table[level+1] = QuadESWeights{T}(weights_n, weights_p)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(table), origin
end

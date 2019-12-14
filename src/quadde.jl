using LinearAlgebra: norm


struct QuadDE{T<:AbstractFloat,N}
    qts::QuadTS{T,N}
    qes::QuadES{T,N}
    qss::QuadSS{T,N}
end
function QuadDE(T::Type{<:AbstractFloat}; maxlevel::Integer=10, kwargs...)
    @assert maxlevel > 0
    qts = QuadTS(T; maxlevel=maxlevel, kwargs...)
    qes = QuadES(T; maxlevel=maxlevel, kwargs...)
    qss = QuadSS(T; maxlevel=maxlevel, kwargs...)
    QuadDE{T,maxlevel+1}(qts, qes, qss)
end

function (q::QuadDE{T,N})(f::Function, a::Real, b::Real;
                          kwargs...) where {T<:AbstractFloat,N}
    if a > b
        I, E = q(f, b, a; kwargs...)
        return -I, E
    end

    if a == b
        return zero(f(a)), zero(T)
    end

    if a == -Inf && b == Inf
        # integrate over [-∞, ∞]
        return q.qss(f; kwargs...)
    elseif b == Inf
        # integrate over [a, ∞]
        if a == 0
            return q.qes(f; kwargs...)
        else
            _a = T(a)
            return q.qes(u -> f(u + _a); kwargs...)
        end
    elseif a == -Inf
        # integrate over [-∞, b]
        if b == 0
            return q.qes(u -> f(-u); kwargs...)
        else
            _b = T(b)
            return q.qes(u -> f(-u + _b); kwargs...)
        end
    else
        # integrate over [a, b]
        return q.qts(f, a, b; kwargs...)
    end
end
function (q::QuadDE{T,N})(f::Function, a::Real, b::Real, c::Real...;
                          atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    bc = (b, c...)
    n = length(bc)
    # FIXME: This anonymous function is not necessarily in principle, but
    #        memory usage increases and execution time gets longer if it is
    #        omitted. I don't know why.
    Ih, E = q(x -> f(x), a, b; atol=atol/n, rtol=rtol/n)
    for i in 2:n
        dIh, dE = q(x -> f(x), bc[i-1], bc[i]; atol=atol/n, rtol=rtol/n)
        Ih += dIh
        E += dE
    end
    Ih, E
end

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
            return q.qes(u -> f(u + T(a)); kwargs...)
        end
    elseif a == -Inf
        # integrate over [-∞, b]
        if b == 0
            return q.qes(u -> f(-u); kwargs...)
        else
            return q.qes(u -> f(-u + T(b)); kwargs...)
        end
    else
        # integrate over [a, b]
        if a == -1 && b == 1
            return q.qts(f; kwargs...)
        else
            _a = T(a)
            _b = T(b)
            s = _b + _a
            t = _b - _a
            Ih, E = q.qts(u -> f((s + t*u)/2); kwargs...)
            return Ih*t/2, E*t/2
        end
    end
end
function (q::QuadDE{T,N})(f::Function, a::Real, b::Real, c::Real...;
                          atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    bc = (b, c...)
    n = length(bc)
    _atol = atol/n
    _rtol = rtol/n
    Ih, E = q(f, a, b; atol=_atol, rtol=_rtol)
    for i in 2:n
        dIh, dE = q(f, bc[i-1], bc[i]; atol=_atol, rtol=_rtol)
        Ih += dIh
        E += dE
    end
    Ih, E
end

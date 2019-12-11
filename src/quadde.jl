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
    if a == b
        zero(f(a)), zero(T)
    elseif a > b
        I, E = q(f, b, a; kwargs...)
        -I, E
    else
        if a == -Inf && b == Inf
            # integrate over [-∞, ∞]
            q.qss(f; kwargs...)
        elseif b == Inf
            if a == 0
                # integrate over [0, ∞]
                q.qes(f; kwargs...)
            else
                # integrate over [a, ∞]
                q.qes(u -> f(u + a); kwargs...)
            end
        elseif a == -Inf
            # integrate over [-∞, b]
            q(u -> f(-u), -b, Inf; kwargs...)
        else
            # integrate over [a, b]
            q.qts(f, a, b; kwargs...)
        end
    end
end
function (q::QuadDE{T,N})(f::Function, a::Real, b::Real, c::Real...;
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

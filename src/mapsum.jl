# License is MIT: https://julialang.org/license
# The code in this file is originally copied from,
#   https://github.com/JuliaLang/julia/blob/2d5741174ce3e6a394010d2e470e4269ca54607f/base/reduce.jl#L147-L168
# and modified by Masaaki Nakamura since `mapreduce_imple()` is required but it
# was not public api.

@noinline function mapsum(f, A::AbstractVector,
                          ifirst::Integer, ilast::Integer, blksize::Int)
    if ifirst == ilast
        @inbounds a1 = A[ifirst]
        return f(a1)
    elseif ifirst + blksize > ilast
        # sequential portion
        @inbounds a1 = A[ifirst]
        @inbounds a2 = A[ifirst+1]
        v = f(a1) + f(a2)
        @simd for i in ifirst + 2 : ilast
            @inbounds ai = A[i]
            v = v + f(ai)
        end
        return v
    else
        # pairwise portion
        imid = (ifirst + ilast) >> 1
        v1 = mapsum(f, A, ifirst, imid, blksize)
        v2 = mapsum(f, A, imid+1, ilast, blksize)
        return v1 + v2
    end
end


"""
    mapsum(f, A::AbstractVector, ifirst::Integer=1, ilast::Integer=length(A))

Return the total summation of items in `A` from `istart`-th through `iend`-th
with applying a function `f`.

NOTE: This function doesn't check `ifirst` and `ilast`. Be careful to use.
"""
mapsum(f, A::AbstractVector, ifirst::Integer=1, ilast::Integer=length(A)) =
    mapsum(f, A, ifirst, ilast, 512)

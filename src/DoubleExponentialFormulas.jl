module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    quadde,
    quadts,
    quades,
    quadss,
    QuadTS,
    QuadES,
    QuadSS

const h0inv = 8

function generate_table end
function trapez end

function quadde end

include("quadts.jl")
function quadts end

include("quades.jl")
function quades end

include("quadss.jl")
function quadss end


end # module

using Documenter, DoubleExponentialFormulas

makedocs(;
    modules=[DoubleExponentialFormulas],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
    repo="https://github.com/machakann/DoubleExponentialFormulas.jl/blob/{commit}{path}#L{line}",
    sitename="DoubleExponentialFormulas.jl",
    authors="Masaaki Nakamura",
    assets=String[],
)

deploydocs(;
    repo="github.com/machakann/DoubleExponentialFormulas.jl",
)

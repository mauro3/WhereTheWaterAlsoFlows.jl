using WhereTheWaterAlsoFlows
using Documenter

makedocs(;
    modules=[WhereTheWaterAlsoFlows],
    authors="Mauro Werder <mauro3@runbox.com> and contributors",
    repo="https://github.com/mauro3/WhereTheWaterAlsoFlows.jl/blob/{commit}{path}#L{line}",
    sitename="WhereTheWaterAlsoFlows.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mauro3.github.io/WhereTheWaterAlsoFlows.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mauro3/WhereTheWaterAlsoFlows.jl",
)

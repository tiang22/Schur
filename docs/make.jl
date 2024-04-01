using icebox
using Documenter

DocMeta.setdocmeta!(icebox, :DocTestSetup, :(using icebox); recursive=true)

makedocs(;
    modules=[icebox],
    authors="Rose-max111",
    sitename="icebox.jl",
    format=Documenter.HTML(;
        canonical="https://Rose-max111.github.io/icebox.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Rose-max111/icebox.jl",
    devbranch="main",
)

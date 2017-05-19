include("../src/EmpiricalBayes.jl")
using ClobberingReload
#creload("../src/EmpiricalBayes.jl")
using EmpiricalBayes
using Distributions
using Base.Test



a = ["pierre", "henri", "annie"]
b = [[1, 1, 2, 1, 1], [1, 3, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 2, 4]]

dc = EmpiricalBayes.DirichletCategorical(Dict(zip(a, b)), 4)
@test isapprox([4, 0, 1, 0], dc.posteriors["henri"].alpha, atol=0.001)

b = [[.34, .5, -.1], [.1, .2, .3, .8], [.4, .5, .7]]
values = Dict(zip(a, b))
nn = EmpiricalBayes.NormalNormal(Dict(zip(a, b)))
@test isapprox(nn.posteriors["henri"].μ, .365, atol = 0.001)

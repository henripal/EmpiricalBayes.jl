using EmpiricalBayes
using Base.Test

# testing DirichletCategorical
names = ["pierre", "henri", "annie"]
categorical_realizations = [[1, 1, 2, 1, 1], [1, 3, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 2, 4]]
counts = Dict(zip(names, categorical_realizations))

dc = EmpiricalBayes.DirichletCategorical(counts, 4)
@test isapprox([4, 0, 1, 0], dc.posteriors["henri"].alpha, atol=0.001)

# testing NormalNormal
value_realizations = [[.34, .5, -.1], [.1, .2, .3, .8], [.4, .5, .7]]
values = Dict(zip(names, value_realizations))
nn = EmpiricalBayes.NormalNormal(values)
@test isapprox(nn.posteriors["henri"].μ, .365, atol = 0.001)

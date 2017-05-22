
module EmpiricalBayes

using DataFrames
using Distributions

export DirichletCategorical,
       NormalNormal,
       dirichlet_from_df,
       normal_from_df

import Base.show

include("common.jl")
include("dirichletcategorical.jl")
include("normalnormal.jl")

end

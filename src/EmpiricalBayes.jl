
module EmpiricalBayes

export fit_bayes, BayesModel

using DataFrames
using Distributions
"""
This is to be used on a dataframe with one `id` column delimiting each individual.
For each `id` there will be several trial rows, containing the result of a
(for now) Categorical distribution.

Step 1: fit the categorical distribution for each `id`, obtaining a distribution of
probabilities.
Step 2: fit a Dirichlet to the distribution of probabilities
Step 3: per id, output the posterior mean of the dirichlet-categorical.
"""
type BayesModel
    n_cat::Int64
    ids::Array{Any, 1}
    prior::Distributions.Dirichlet{Float64}
    likelihoods::Array{Distributions.Categorical{Float64}, 1}
    probabilities::Array{Float64, 2}
end
"""
Replace the zeros with epsilon values in the probability vector so the dirichlet
can be fit?
This is an awful hack which I should fix ASAP
"""
function zero_out(p::Array{Float64, 1})
    ϵ = .001
    isapprox(sum(p), 1.0)  || error("Probabilities do not add to 1")
    p += ϵ
    return p/sum(p)
end

"""
Fit the model to the dataframe
"""
function fit_bayes(df::DataFrames.DataFrame, id::Symbol, category::Symbol )
    unique_ids = unique(df[id])
    unique_categories = unique(df[category])

    n_ids = length(unique_ids)
    n_cat = length(unique_categories)

    likelihoods = Array{Distributions.Categorical{Float64}, 1}()
    probabilities = Array{Float64, 2}(n_cat, n_ids)

    for i in 1:n_ids
        categories = df[df[id] .== unique_ids[i], :][category]
        fitted = fit(Categorical, n_cat, categories)
        push!(likelihoods, fitted)
        probabilities[:, i] = zero_out(fitted.p)
    end

    prior = fit_mle(Dirichlet, probabilities)

    return BayesModel(n_cat, unique_ids, prior, likelihoods, probabilities)
end

function return_posterior(df::DataFrame, id::Symbol, category::Symbol, bm::BayesModel)
    aggr_df = by(df, [id, category], df->DataFrame(N = size(df, 1)))
    aggr_df_wide = unstack(aggr_df, id, category, :N)
    for (colname, colval) in eachcol(aggr_df_wide[:, 2:size(aggr_df_wide)[2]])
        aggr_df_wide[colname] = convert(Array, colval, 0)
    end

    aggr_df_wide[:adjusted_average] = 0.0

    # this is where the shrinking happens
    adjusted_average = Matrix(aggr_df_wide[2:(bm.n_cat + 1)]) .+ bm.prior.alpha'
    mean = Array{Float64}(size(adjusted_average)[1])

    for i in 1:size(adjusted_average)[1]
        row = adjusted_average[i, :]
        row = row/sum(row)
        aggr_df_wide[i, :adjusted_average] = (row' * collect(1:bm.n_cat))[1]
    end

    return aggr_df_wide
end


end

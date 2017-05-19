
module EmpiricalBayes

export DirichletCategorical, NormalNormal, dirichlet_df

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
abstract BayesModel{S}

type DirichletCategorical{S} <: BayesModel
    sample_sizes::Dict{S, Int64}
    prior::Distributions.Dirichlet{Float64}
    likelihoods::Dict{S, Distributions.Categorical{Float64}}
    posteriors::Dict{S, Distributions.Dirichlet{Float64}}
end

type NormalNormal{S} <: BayesModel
    sample_sizes::Dict{S, Int64}
    prior::Distributions.Normal{Float64}
    likelihoods::Dict{S, Distributions.Normal{Float64}}
    posteriors::Dict{S, Distributions.Normal{Float64}}
end

function NormalNormal{S}(values::Dict{S, Array{Float64, 1}})
    sample_sizes = Dict{S, Int64}()
    likelihoods = Dict{S, Distributions.Normal{Float64}}()
    posteriors = Dict{S, Distributions.Normal{Float64}}()
    means = zeros(length(values))

    # this is awful. 
    all_values = Float64[]
    for (_, value) in values
       all_values = vcat(all_values, value) 
    end

    σ = sqrt(var(all_values)) #n-1?

    for (i, (name, value)) in enumerate(values)
        sample_sizes[name] = length(value)
        likelihoods[name] = fit(Normal, value)
        means[i] = likelihoods[name].μ
    end

    prior = fit(Normal, means)

    for (name, value) in values
        # conjugate magic
        n = sample_sizes[name]
        posterior_variance = 1/(1/prior.σ^2 + n/σ^2)
        posterior_mean = (prior.μ/prior.σ^2 + sum(value)/σ^2)*posterior_variance
        posteriors[name] = Normal(posterior_mean, sqrt(posterior_variance))
    end

    NormalNormal(sample_sizes, prior, likelihoods, posteriors)
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
Fit the dirichlet categorical with an empirical prior
Return the fitted Dirichlet Categorical object
counts is a dictionary indexed by any type of key whose values are
arrays of counts [1, 2, 2, 1, ...]
n_categories is the number of distinct categories
"""
function DirichletCategorical{S}(counts::Dict{S, Array{Int64, 1}}, n_categories::Int64)
    sample_sizes = Dict{S, Int64}()
    likelihoods = Dict{S, Distributions.Categorical{Float64}}()
    posteriors = Dict{S, Distributions.Dirichlet{Float64}}()
    P = zeros(length(counts), n_categories)

    for (i, (name, count)) in enumerate(counts)
        sample_sizes[name] = length(count)
        likelihoods[name] = fit(Categorical, n_categories, count) 
        P[i, :] = zero_out(likelihoods[name].p)
    end

    prior = fit(Dirichlet, P')

    for (name, count) in counts
        # conjugate magic
        posteriors[name] = Dirichlet(likelihoods[name].p * sample_sizes[name] + prior.alpha)
    end

    DirichletCategorical(sample_sizes, prior, likelihoods, posteriors)
end

        
"""
Fit the model to the dataframe.
This needs df[df[:id] .== id, :][:category] to be an integer vector between
1 and n_max
"""
function dirichlet_df(df::DataFrames.DataFrame, id::Symbol, category::Symbol )
    unique_ids = unique(df[id])
    unique_categories = unique(df[category])

    n_ids = length(unique_ids)
    n_cat = length(unique_categories)

    counts = Dict{typeof(unique_ids[1]), Array{Int64, 1}}()

    for name in unique_ids
        categories = df[df[id] .== name, :][category]
        counts[name] = categories
    end

    return DirichletCategorical(counts, n_cat)
end



end

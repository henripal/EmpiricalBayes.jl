
"""
Contains samples, priors, likelihoods and posteriors.
Is updated and fitted to data upon instanciation.
"""
abstract BayesModel{S}


    
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

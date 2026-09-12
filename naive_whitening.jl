using LinearAlgebra, Statistics

function naive_estimate(S′, ε, τ=0.1, k=nothing; iters=nothing, limit=nothing)
    # No trimming — just the empirical mean/covariance of ALL points
    n = size(S′, 2)
    return trues(n)  # "select everyone" -> whitening uses the raw, contaminated covariance
end
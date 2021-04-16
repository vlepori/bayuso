# Option A: Multiple regression model
# with hyperpriors

log_pdf_exp(x,lambda) = log(lambda) - lambda*x    

import CSV, DataFrames, Plots
data_t = CSV.File(read("exercises/bayuso/co2_ozone_data.txt")) |> DataFrames.DataFrame
#data[:,1]=data[:,1].+1880
Plots.plot(data[:,2],data_t[:,3])

# mu_t = mu_0 + a * co2 + b * ozone 
# T_t  ~ N(mu_t, sigma)
# priors :
# mu_0 ~ unif(-Inf, Inf)
# a,b ~ N(0, sigma_ab)
# sigma_ab ~ Exp(0.5)
# sigma ~ gamma(2,.02)

temps = data[:,2]
predictors = data_t[:,2:3]

#init = [13.5, 0, 0, .1, .1] # mu0, a, b, sigma_ab, sigma

function lik4(theta)
    mu_ts = theta[1] .+ theta[2]*predictors[:,1] .+ theta[3]*predictors[:,2]
    return sum(log_pdf_normal.(temps,mu_ts,theta[5]))
end
    
function prior4(theta)
    prior_mu_zero = 0
    prior_a = log_pdf_normal(theta[2],0,theta[4])
    prior_b = log_pdf_normal(theta[3],0,theta[4]) 
    prior_sigma_ab = log_pdf_exp(theta[4], 0.5) # lambda = 0.5
    prior_sigma = log_pdf_gamma(theta[5],2,.02)
    return prior_mu_zero+prior_a+prior_b+prior_sigma+prior_sigma_ab
end

function kernels4(theta)
    n_mu0 = sliding_window(theta[1], 1)
    n_a = sliding_window(theta[2],.1)
    n_b = sliding_window(theta[3],.05)
    n_sab = abs(sliding_window(theta[4],1))
    n_sigma = abs(sliding_window(theta[5], .05))
    return [n_mu0, n_a, n_b, n_sab, n_sigma]
end

init = [13.5, 0, 0, .1, .1] # mu0, a, b, sigma_ab, sigma


test_thetas, test_probs = sample_mcmc(init, kernels4, prior4, lik4, 10000000, se=1000 )

plot(test_thetas[5:end,:], layout = (5, 1), legend=false, title=["mu0" "a" "b" "sig_ab" "sigma"])
plot!(size=(1000,1000))
savefig("Ex4.png")

plot(test_probs[5:end,:], layout = (3, 1), legend=false, title=["log prior" "log likelihood" "log unnormalized posterior"])
histogram(test_thetas[:,4],legend=false, title = "sab")



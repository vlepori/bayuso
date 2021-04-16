import Statistics
using Plots

include("pdfs.jl")
include("mcmc.jl")

bears_bm = [67.65, 92.13, 58.92, 87.64, 76.31, 88.86]
pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]

# bm_j ~ N(mu_j, sigma_j) , j in {b,p}
# mu_b, mu_p ~ U(-Inf,Inf)
# sigma_b, sigma_p ~ gamma(alpha, beta)
# alpha, beta ~ Exp(0.1)

# theta: [mu_panda,mu_bear,sigma_panda,sigma_bear,alpha,beta]

function lik4b(theta)
    lik_p = sum(log_pdf_normal.(pandas_bm,theta[1],theta[3]))
    lik_b = sum(log_pdf_normal.(bears_bm,theta[2],theta[4]) )
    return lik_p+lik_b
end
    
function prior4b(theta)
    p_sig_p = log_pdf_gamma(theta[3],theta[5],theta[6])
    p_sig_b = log_pdf_gamma(theta[4],theta[5],theta[6])
    p_alpha = log_pdf_exp(theta[5],0.1)
    p_beta  = log_pdf_exp(theta[6],0.1)
    return 0+0+ p_sig_p + p_sig_b + p_alpha + p_beta
end

function kernels4b(theta)
    n_mup = sliding_window(theta[1], 4)
    n_mub = sliding_window(theta[2], 4)
    n_sigp = abs(sliding_window(theta[3],3))
    n_sigb = abs(sliding_window(theta[4],3))
    n_alpha = abs(sliding_window(theta[5], 5))
    n_beta = abs(sliding_window(theta[6], 5))
    return [n_mup, n_mub, n_sigp, n_sigb, n_alpha, n_beta]
end

init = [50.0,50.0,3.0,3.0,1,10] # mu0, a, b, sigma_ab, sigma

test_thetas, test_probs = sample_mcmc(init, kernels4b, prior4b, lik4b, 10000000, se=1000 )

plot(test_thetas[5:end,:], layout = (6, 1), legend=false, title=["mu_panda" "mu_bear" "sig_pd" "sig_br" "alpha" "beta"])
plot!(size=(1400,1200))
plot(test_probs[5:end,:], layout = (3, 1), legend=false, title=["log prior" "log likelihood" "log unnormalized posterior"])
savefig("Ex4b.png")

histogram(test_thetas[:,1]-test_thetas[:,2],legend=false, title = "delta_mu")
Statistics.quantile(test_thetas[:,1]-test_thetas[:,2],[0.01,0.99])
histogram(test_thetas[:,3]-test_thetas[:,4],legend=false, title = "delta_sigma")

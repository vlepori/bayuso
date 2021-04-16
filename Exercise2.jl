using Plots
import SpecialFunctions
log_pdf_normal(x,mu,sigma) = -.5*log(2pi*sigma^2) -((x-mu)^2/(2*sigma^2))
log_pdf_gamma(x,al,bt) = al*log(bt) - log(SpecialFunctions.gamma(al)) + (al - 1)*log(x) - bt*x
sliding_window(x::Float64, width) = x+2*(rand()-.5)*width
pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]

function sample_mcmc(theta_init::Array{Float64}, proposal_f::Function, prior_f::Function, lik_f::Function, iter::Int; se::Int = 1)
    thetas = zeros(Int(iter/se+1),length(theta_init))
    thetas[1,:] = theta_init
    cur_theta = theta_init

    cur_prior = prior_f(cur_theta)
    cur_lik = lik_f(cur_theta)
    cur_post = cur_lik+cur_prior
    plps = zeros(Int(iter/se+1),3)
    plps[1,:] = [cur_prior,cur_lik,cur_post]

    for i in 2:iter
        new_theta = proposal_f(cur_theta)
        new_lik = lik_f(new_theta)
        new_prior = prior_f(new_theta)
        new_post = new_lik+new_prior
        r = exp(new_post-cur_post)
        (rand()<r) && (cur_theta = new_theta; cur_prior = new_prior; cur_lik = new_lik; cur_post = new_post)
        if (i % se == 0)
            thetas[Int(i/se+1),:] = cur_theta
            plps[Int(i/se+1),:] = [cur_prior,cur_lik,cur_post]
        end
    end
    return thetas, plps
end

# run
normal_lik(x) = sum(log_pdf_normal.(pandas_bm,x[1],x[2]))
u_kernels(x) = [(sliding_window(x[1],10)),abs(sliding_window(x[2],10))]
gamma_prior(x) = 0+log_pdf_gamma(x[2],2,.02)
n_iter=10000
test_thetas, test_probs = sample_mcmc([100.0,20.0], u_kernels, gamma_prior, normal_lik, n_iter )
plot(1:n_iter, test_thetas, layout = (2, 1), legend=false, title=["mu" "sigma"])
plot(1:n_iter, test_probs, layout = (3, 1), legend=false, title=["log prior" "log likelihood" "log unnormalized posterior"])

import DataFrames, CSV
df = DataFrames.DataFrame(hcat(test_probs,test_thetas))
DataFrames.rename!(df, ["mu","sigma","prior","lik","post"])
CSV.write("trace.tsv", df; delim = '\t')
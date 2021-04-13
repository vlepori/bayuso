using Plots
import SpecialFunctions
log_pdf_normal(x,mu,sigma) = -.5*log(2pi*sigma^2) -((x-mu)^2/(2*sigma^2))
log_pdf_gamma(x,al,bt) = al*log(bt) - log(SpecialFunctions.gamma(al)) + (al - 1)*log(x) - bt*x

sliding_window(x::Float64, width) = x+2*(rand()-.5)*width
pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]

function sample_mcmc(theta_init::Array{Float64}, proposal_f::Function, prior_f::Function, lik_f::Function, iter::Int, log_f::Bool = true)
    thetas = zeros(iter,length(theta_init))
    thetas[1,:] = theta_init
    current_theta = theta_init
    for i in 2:iter
        new_theta = proposal_f(current_theta)
        if log_f
            r = exp((prior_f(new_theta)+lik_f(new_theta)) - (prior_f(current_theta)+lik_f(current_theta)))
        else
            r = (prior_f(new_theta)*lik_f(new_theta))/(prior_f(current_theta)*lik_f(current_theta))
        end
        (rand()<r) && (current_theta = new_theta)
        thetas[i,:] = current_theta
    end
    return thetas
end

# estimate mu and sigma
normal_lik(x) = sum(log_pdf_normal.(pandas_bm,x[1],x[2]))
u_kernels(x) = [(sliding_window(x[1],10)),abs(sliding_window(x[2],10))]
gamma_prior(x) = 0+log_pdf_gamma(x[2],2,.02)
n_iter=10000
test = sample_mcmc([50.0,1.0], u_kernels, gamma_prior, normal_lik, n_iter )
plot(1:n_iter, test, layout = (2, 1), legend=false, title=["mu" "sigma"])
# SYMMETRIC UNIFORM PROPOSAL
sliding_window(x::Float64, width) = x+2*(rand()-.5)*width


# SIMPLE MCMC
# Array of initial values for theta :: Array
# fun to propose updates for theta :: Function
# fun to calcultate prior for given theta :: Function
# fun to calculate lik of data for given theta :: Function
# iterations to run :: Int
# save every :: Int
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

# Run multiple chains in parallel
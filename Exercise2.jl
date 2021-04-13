sliding_window(x::Float64, width) = x+2*(rand()-.5)*width

function sample_mcmc(theta_init::Float64, proposal_f::Function, prior_f::Function, lik_f::Function, iter, log_f = true)
    # thetas = [theta_init]
    thetas = zeros(iter,length(theta_init))
    thetas[1] = theta_init
    #print(thetas)
    current_theta = theta_init
    for i in 2:iter
        new_theta = proposal_f(current_theta)
        if log_f
            r = exp(prior_f(new_theta)*lik_f(new_theta) - prior_f(current_theta)*lik_f(current_theta))
        else
            r = (prior_f(new_theta)*lik_f(new_theta))/(prior_f(current_theta)*lik_f(current_theta))
        end
        (rand()<r) && (current_theta = new_theta)
        #push!(thetas, current_theta)
        thetas[i] = current_theta
    end
    return thetas
end

# fixed sigma
test = sample_mcmc(50.0, x->abs(sliding_window(x,5)), x->1, x-> sum(log_pdf_normal.(pandas_bm,x,12)), 10000, true)

test = sample_mcmc([50.0,10.0], x->abs(sliding_window(x,5)), x->1, x-> sum(log_pdf_normal.(pandas_bm,x,12)), 10000, true)
plot(test,legend=false)
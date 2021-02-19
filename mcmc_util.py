from scipy.fftpack import fft, ifft, ifftshift
import numpy as np

def autocorrelation(x):
    """
    compute the autocorrelogram of a time series using FFT
    
    args:
    ====
     x: uni-dimensional time series as numpy array
     
    out:
    ===
     numpy vector describing the autocorrelogam
    """
    xp = ifftshift((x - np.mean(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def normalized_ess(x, threshold = 0.3):
    """
    Computes the Effective Sample Size by fiting an exponential 
    to the autocorrelation (i.e. autocorrelation from an AR(1) process)
    
    This is done by fitting a line to [k, log(autocorrelation[k])]
    
    args:
    ====
     x: scalar time series
     threshold: estimate the autocorrelation coefficient that are above that threshold
    
    out:
    ===
        {
         "lambda_hat": rate of decay of autocorrelation
         "IACT": integrated autocorrelation
         "ESS_normalized": inverse IACT
        }
    
    """
    autocorr = autocorrelation(x)
    k_max = np.argmax(autocorr < threshold)
    if k_max == 0:
        print("normalized_ess:: short time series")
        k_max = np.sqrt(len(x)).astype(int)
        
    x_ = np.arange(k_max)
    y_ = np.log(autocorr[:k_max])
    lambda_hat = -np.sum(x_ * y_) / np.sum(x_ * x_)
    IACT = 2./(1-np.exp(-lambda_hat)) - 1
    ESS_normalized = 1. / IACT
    return {"lambda_hat": lambda_hat,
            "IACT": IACT,
            "ESS_normalized": ESS_normalized}


def brownian_update(v_unit, dt):
    """
    args:
    ====
     v_unit: unit vector
     dt: time increment
     
    out:
    ===
     v_new: vector distributed as P(v_unit, dt) where P is the dynamic of a Brownian on the unit sphere
    """
    d = len(v_unit)
    perturb = np.random.normal(size=d) / np.sqrt(d)
    alpha = np.exp(-dt/2.)
    v_new = alpha * v_unit + np.sqrt(1-alpha**2)*perturb
    return v_new / np.linalg.norm(v_new)


def bounce(v, grad_vector):
    """
    elastic bounce
    
    args:
    ====
      v: input velocity
      grad_vector: a vector discribing the hyperplane
    
    """
    normalized_grad = grad_vector
    normalized_grad = normalized_grad / np.linalg.norm(normalized_grad)
    return -v + 2* np.dot(v, normalized_grad)*normalized_grad

def DBPS(param):
    """
    run a Discrete Bouncy Particle Sampler
    
    args:
    ====
    param: dictionary of parameters
        param["x_init"]: initial position
        param["v_init"]: initial velocity -- unit vector
        param["n_mcmc"]: number of steps
        param["delta"]: spatial discretization
        param["kappa"]: refreshment rate
        param["log_target"]: log_target function
        param["grad_log_target"]: grad_log_target function
        param["thinning"]: thinning parameter -- equals 1 by default
        param["verbose"]: True/False -- equals False by default
    
    out:
    ===
    mcmc_out dictionary
        {"log_posterior" : loglik_history
         "mean_dot_product" : mean_dot_product,
         "acceptance_rate" : acceptance_rate, 
         "DR_acceptance_rate" : DR_acceptance_rate}
    """
    
    # get parameters
    x_init = param["x_init"]
    v_init = param["v_init"]
    n_mcmc = param["n_mcmc"]
    delta = param["delta"]
    kappa = param["kappa"]
    F = param["log_target"]
    grad_F = param["grad_log_target"]
    thinning = param["thinning"] if "thinning" in param.keys() else 1
    verbose = param["verbose"] if "verbose" in param.keys() else False
    

    #initialization
    dim = len(x_init)
    loglik_history = np.zeros(n_mcmc // thinning)    
    x = np.copy(x_init)
    v = np.copy(v_init)
    v = v / np.linalg.norm(v)
    direction_after_bounce = np.copy(v)
    
    log_target_current = F(x)
    acceptance_rate = 0
    DR_acceptance_rate = 0
    DR_attempts = 0
    

    nb_bounce = -1
    mean_dot_product = 0.
    for k in range(n_mcmc):
        if verbose and k % (n_mcmc // 10) == 0:
            print("Sampler: delta={} \t kappa={} \t Work Done={} ".format(delta, kappa, 100*k / float(n_mcmc)))
            
        # update position
        x_new, v_new = x + delta * v, -v
        log_target_new = F(x_new)
        accept_1 = min(1, np.exp(log_target_new - log_target_current))
        
        #accept/reject
        if np.random.rand() < accept_1:
            acceptance_rate += 1.0
            x, v = x_new, v_new
            log_target_current = log_target_new
            
        #if move rejected, attempt a bounce
        else:
            nb_bounce += 1
            mean_dot_product += np.dot(v, direction_after_bounce)
            
            DR_attempts += 1.
            
            #attempt delayed-rejection
            grad_vector = grad_F(x_new)
            v_bounced = bounce(v_new, grad_vector)
            
            
            x_bounced, v_bounced = x_new + delta * v_bounced, -v_bounced
            
            log_target_bounced = F(x_bounced)
            accept_2 = min(1, np.exp(log_target_new - log_target_bounced))
            if np.random.rand() < ( (1-accept_2) / (1-accept_1) ) * np.exp(log_target_bounced - log_target_current):
                DR_acceptance_rate += 1.
                x, v = x_bounced, v_bounced
                log_target_current = log_target_bounced
                
            #direction_at_bouncing_after[nb_bounce,:] = -v
            direction_after_bounce = -v
    
        #flip
        v = -v
        
        # Brownian perturbation
        v = v / np.linalg.norm(v)
        v = brownian_update(v, delta*kappa)
        
        if k % thinning == 0:
            loglik_history[k // thinning] = log_target_current #F(x)
                
    if DR_attempts > 0:
        DR_acceptance_rate = DR_acceptance_rate / float(DR_attempts)
    acceptance_rate = acceptance_rate / n_mcmc
    
    if nb_bounce > 0:
        mean_dot_product = mean_dot_product / float(nb_bounce)
    else:
        mean_dot_product = 1.
    
    return {"log_posterior" : loglik_history,
            "mean_dot_product" : mean_dot_product,
            "acceptance_rate" : acceptance_rate, 
            "DR_acceptance_rate" : DR_acceptance_rate,
            "steps_between_bounces": n_mcmc / float(nb_bounce +1.) }





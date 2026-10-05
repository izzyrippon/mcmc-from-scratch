import pandas as pd
import numpy as np
import math as m
import scipy.stats as sp
import statistics as stats
import random
import matplotlib.pyplot as plt

# Toy observed data to demonstrate the MCMC sampling process
t = list(range(18))
y = [5.14, 6.38, 5.92, 6.88, 7.29, 6.12, 7.90, 7.88, 8.12, 8.29, 7.38, 8.50, 8.62, 8.75, 8.68, 9.00, 9.12, 9.06]

# Plot the observed data
plt.scatter(t,y)

##########################################################
# Independance sample model
##########################################################

def ind_model(nsamples, thetaInit, propMean, propCov, t, y):
  output = np.zeros((nsamples, 3))
  theta = thetaInit
  acc = 0
  for i in range(1, nsamples+1):
    theta_p = np.random.multivariate_normal(propMean, propCov)
    if (theta_p[0]>0):
        w = sum(np.square([y[x]-theta[0]*1-(np.exp(-x*np.exp(theta[1]))) for x in t]))**(-8)/sp.multivariate_normal.pdf(theta, mean=propMean, cov=propCov)
        wp = sum(np.square([y[x]-theta_p[0]*1-(np.exp(-x*np.exp(theta_p[1]))) for x in t]))**(-8)/sp.multivariate_normal.pdf(theta_p, mean=propMean, cov=propCov)
        u = random.uniform(0, 1)
        if(u<min(wp/w,1)):
          theta = theta_p
          acc = acc+1
          
    output[i-1,:] = [theta[0],theta[1],acc/i]  # theta should be a vector of length 2,
    # also record the acceptance rate for tuning purposes
  return output

# Choosing covariance for optimal exploration, aiming for around 0.25-0.5 acceptance rate
# 
accepted = np.zeros((4,4))
tests = [0.05, 0.5, 5, 50]
mu = [7, 0]
for i in range(1, 5):
  for j in range(1, 5):
    Sigma = np.array([[tests[i-1], 0], [0, tests[j-1]]])
    accepted[i-1, j-1] = ind_model(500, [7, 0], mu, Sigma, t, y)[499, 2]

accepted

# Choose covariance [[5, 0], [0, 0.5]] which gives an acceptance rate of 0.256
ind_samples = ind_model(10000, [7, 0], mu, np.array([[0.5, 0], [0, 0.5]]), t, y)

# Plot burn in
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)    
plt.plot(ind_samples[:, 0])
plt.title('Trace plot for theta_0')
plt.subplot(1, 2, 2)
plt.plot(ind_samples[:, 1])
plt.title('Trace plot for theta_1')
plt.show()

# Plot parameter distributions
x_eval = np.linspace(0, 0.3, num=200) 
y = sp.uniform.pdf(x_eval, 0, 0.2) 
x_eval2 = np.linspace(0, 3.5, num=200) 
y2 = sp.uniform.pdf(x_eval, 0, 3) 
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Parameter distributions')
ax1.hist(ind_samples[:, 0],bins=200,density=True,histtype="step", label='E1')
ax1.set(xlabel='$\lambda$')
ax1.legend()
ax2.hist(ind_samples[:, 1],bins = 200,density=True,histtype="step", label='E1')
ax2.set(xlabel='C')
ax2.legend()

##########################################################
# RWMH model
##########################################################

# Function to simulate data points from a given function 
def sim_dat(f,theta,times):
    
    res = [f(i,theta) for i in times]
    return res

# Function to calculate the log-likelihood of observed values given the model and parameters
# (Chi-Squared likelihood)
def loglikest(f,theta, times, obsdata,sd):
    
    test = sim_dat(f, theta, times)
    # assumes gaussian likelihood
    if theta[0]>0 and theta[1]>0:
        s = [(obsdata[i]-test[i])**2/sd**2 + np.log(2*m.pi*sd**2) for i in times]
        ll = -1/2*sum(s)
    else:
        ll = -500000
    return ll

# Function to perform MCMC sampling for parameter estimation
def my_mcmc(f,obsdata,times):
    
    all_theta = []
    
    # initialise theta
    theta0 = [random.uniform(0,1),random.uniform(0,3)]
    Aset = [theta0[0]]
    Cset = [theta0[1]]
    lset = [loglikest(f,theta0,times,obsdata,0.1)]

    for i in range(10000):
        
        # random step to propose new theta
        theta = [Aset[-1]+np.random.normal(0,0.05),Cset[-1]+np.random.normal(0,0.05)]
        lt = loglikest(f,theta,times,obsdata,0.1)
        
        # accept new theta with acceptance probability
        if lt>lset[-1]+ m.log(random.uniform(0,1)):
            lset.append(lt)
            Aset.append(theta[0])
            Cset.append(theta[1])
        else:
            lset.append(lset[-1])
            Aset.append(Aset[-1])
            Cset.append(Cset[-1])
            
    all_theta.append([Aset,Cset,lset])
    return all_theta


# Define the model function to be fitted to the observed data
def f(t,theta):
    A=theta[0]
    C=theta[1]
    y=C*(1-m.exp(-(A)*t))
    return y

results = my_mcmc(f,y,t)

# To-Do: Plot the trace plots for the parameters A and C to visualize the sampling process

# Plot parameter distributions
x_eval = np.linspace(0, 0.3, num=200) 
y = sp.uniform.pdf(x_eval, 0, 0.2) 
x_eval2 = np.linspace(0, 3.5, num=200) 
y2 = sp.uniform.pdf(x_eval, 0, 3) 
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Parameter distributions')
ax1.hist(results[0][0],bins=200,density=True,histtype="step", label='E1')
ax1.set(xlabel='$\lambda$')
ax1.legend()
ax2.hist(results[0][1],bins = 200,density=True,histtype="step", label='E1')
ax2.set(xlabel='C')
ax2.legend()
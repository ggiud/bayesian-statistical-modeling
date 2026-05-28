// in the data block we can also parse the 
// hyperparameters of the model

data
{
	int<lower = 0> n;       
	int<lower = 0> p;       
	int<lower = 0, upper = 1> y[n];   	
	matrix[n,p] X;		
	
	vector[p] beta0;
	matrix[p,p] Sigma0;
}

// here we have the parameters on which we set a prior distributional assumption

parameters
{
	vector[p] beta;
}

// a possible strategy is to produce as transformation 
// the linear predictor for each observation

transformed parameters 
{
	vector[n] mu;
  mu = 1 - exp(-exp(X * beta));;  // link della cloglog
}

// in the model block we specify the distributional assumption for the prior
// and the likelihood

model
{
	// Prior:
  beta ~ multi_normal(beta0, Sigma0);
	
	// Likelihood:
	y ~ bernoulli(mu);
}

generated quantities 
{
  vector[n] log_lik;
  for (l in 1:n) 
	{
    log_lik[l] = bernoulli_lpmf(y[l] | mu[l]);
  }
}
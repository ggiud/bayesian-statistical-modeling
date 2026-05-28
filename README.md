# 📊 Bayesian Model Averaging for Binary Classification

## **Overview**

This project investigates the role of **model uncertainty** in statistical inference through the application of **Bayesian Model Averaging (BMA)** for binary classification problems.

Instead of selecting a single “best” model, BMA combines multiple candidate models by weighting them according to their posterior probabilities, producing more robust and calibrated predictive inference.

The project includes:

* Bayesian binary regression models
* comparison of different link functions
* Markov Chain Monte Carlo (MCMC)
* Bayesian Model Averaging (BMA)
* posterior inclusion probabilities (PIP)
* predictive uncertainty analysis
* comparison with frequentist ensemble methods (Bagging)

Applications were performed on:

* simulated data
* the real-world **Pima Indians Diabetes Dataset**

---

## **Objectives**

The main goals of the project are:

* quantify model uncertainty
* compare alternative Bayesian binary regression models
* evaluate predictive performance under different link functions
* combine models through Bayesian Model Averaging
* compare Bayesian ensembles with Bagging approaches

---

# **Bayesian Model Averaging (BMA)**

Classical model selection assumes that a single model is correct, ignoring uncertainty regarding the model structure itself.

Bayesian Model Averaging addresses this limitation by combining all candidate models:

[
p(Y|X,D) = \sum_k P(M_k|D), p(Y|X,M_k,D)
]

where:

* (P(M_k|D)) are posterior model probabilities
* (p(Y|X,M_k,D)) are posterior predictive distributions

This framework propagates:

* parameter uncertainty
* structural model uncertainty

leading to more reliable inference and predictive intervals.

---

# **Simulated Data Application**

A simulated dataset was generated with:

* **200 observations**
* **6 predictors**
* correlated covariates
* binary response variable

### **Data Generation**

The first four predictors were sampled independently from:

[
X_i \sim \mathcal{N}(0,1)
]

while the remaining variables were constructed as linear combinations of the first predictors plus Gaussian noise, introducing multicollinearity.

The binary response variable was generated through a logistic model.

---

## **Bayesian Binary Regression Models**

Three Bayesian GLMs were estimated:

### **Logit**

[
p = \frac{1}{1+\exp(-\eta)}
]

### **Probit**

[
p = \Phi(\eta)
]

### **Complementary Log-Log (cloglog)**

[
p = 1-\exp(-\exp(\eta))
]

Each model was implemented separately in **Stan**.

---

## **Prior Specification**

Regression coefficients were assigned a multivariate Gaussian prior:

[
\beta \sim \mathcal{N}_p(\beta_0, \Sigma_0)
]

This choice:

* regularizes estimates
* stabilizes inference under correlated predictors
* improves MCMC efficiency

---

## **Model Space**

The candidate model space included:

* **3 link functions**
* **64 covariate subsets**

for a total of:

[
3 \times 2^6 = 192 \text{ models}
]

Each model was estimated using **Markov Chain Monte Carlo (MCMC)**.

---

# **Results on Simulated Data**

### **Coverage & Predictive Intervals**

The BMA model achieved:

* substantially higher coverage
* wider but more calibrated predictive intervals

compared to:

* null models
* full models
* “correct” models
* best single-link models

### **Key Findings**

* logit and probit produced similar behavior
* cloglog showed lower coverage
* the highest posterior weight was assigned to a logit specification

This result is coherent with the data-generating mechanism.

---

## **Variance Decomposition**

The posterior variance under BMA decomposes into:

[
Var_{BMA}(\beta)=E_M[Var(\beta|M,y)] + Var_M(E[\beta|M,y])
]

capturing:

* within-model uncertainty
* between-model uncertainty

This explains why BMA intervals are wider but more statistically honest.

---

## **Posterior Inclusion Probabilities (PIP)**

Posterior Inclusion Probabilities were computed for all predictors.

The analysis highlighted:

* strong importance of the true generating variables
* uncertainty concentration on correlated covariates
* coherent variable selection behavior

Posterior distributions were visualized through:

* MCMC densities
* aggregated BMA densities
* spike-and-slab inclusion probabilities

---

# **Real Data Application – Pima Dataset**

The methodology was applied to the **Pima Indians Diabetes Dataset**.

The same:

* priors
* link functions
* Bayesian framework

were maintained to evaluate generalization on real data.

---

## **Results**

### **Predictive Performance**

The BMA model achieved:

* predictive accuracy comparable to the best individual model
* significantly higher coverage
* well-calibrated predictive distributions

### **Model Comparison**

The cloglog specification showed:

* lower adaptability
* lower predictive performance

while:

* logit
* probit

performed similarly and substantially better.

---

## **WAIC & LPML**

Model comparison metrics included:

* WAIC
* LPML

BMA showed slightly higher complexity costs due to averaging across heterogeneous link functions, while maintaining strong predictive capability.

---

# **Comparison with Bagging**

As a frequentist benchmark, the project compared BMA with:

* Bootstrap Aggregating (Bagging)
* decision trees

### **Variable Importance**

Both approaches identified:

* glucose
* age
* BMI

as the most influential predictors.

---

## **ROC Curve Analysis**

### **AUC Scores**

* **BMA:** 0.865
* **Bagging:** 0.807

The superior AUC achieved by BMA suggests:

* stronger discriminative capability
* better calibrated predictive probabilities

---

# **Conclusion**

This project demonstrates the effectiveness of Bayesian Model Averaging for handling model uncertainty in binary classification problems.

Main findings include:

* improved predictive calibration
* more realistic uncertainty quantification
* strong robustness across link functions
* superior discriminative performance compared to Bagging

The results highlight how averaging across plausible models can produce more reliable inference than relying on a single selected specification.

---

# **Methods & Techniques**

* Bayesian Model Averaging (BMA)
* Bayesian GLMs
* Logistic Regression
* Probit Regression
* Complementary Log-Log Models
* Markov Chain Monte Carlo (MCMC)
* Stan
* Posterior Inclusion Probabilities (PIP)
* WAIC
* LPML
* ROC Curve Analysis
* Bagging

---

# **Tech Stack**

* **R**
* **Stan**
* **rstan**
* **ggplot2**
* **caret**
* **pROC**


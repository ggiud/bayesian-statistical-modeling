########################################################################### 
# Giuditta ADEZIO, Mattia OTTOLENGHI
# PROGETTO BAYESIAN STATISTICAL MODELING 2026
###########################################################################

library(ggplot2)
library(rstan)
library(reshape2)
library(bridgesampling)
library(MASS)
library(randomForest)
library(pROC)

# WORK IN PROGRESS --------------------------------------------------------

#1) out_coef_values: controllo se è giusto fare così con logit (FATTO)
#2) mi piacerebbe far vedere l'effetto della scelta di prior diverse
#3) capire varianza codice prof, cosa serve?
#4) confrontare previsioni con altri modelli confrontati in precedenza (FATTO)
#5) aggiungere linea rossa per densità coef BMA (vedi sotto, WIP)
#6) CAPIRE COME USARE ANCHE PROBIT, CLOG,...(FATTO)

#a) finire coverage per 3 tipi link (FATTO)
#b) pima dataset reali (FATTO)
#c) bagging (FATTO)
#d) tipo di prior diversa
#e) manca LPML nel ciclo con tre link
#f) out_coef_dens cosa farne?
#g) bayes factor aggiungo per sim  - > spiego come mod migliore <> mod dati generati

# controllare scrittura wei, WAIC, logml



# SIMULAZIONE DATI ---------------------------------------------------------


# creo dataset con variabili correlate simile al paper
set.seed(123)
n_train <- 200
n_test <- 50
p <- 6

# TRAINING
# Predittori indipendenti X1,...,X4 ~ N(0,1)
X_iid <- matrix(rnorm(n_train * 4), nrow = n_train, ncol = 4)
# Predittori correlati X5, X6 con X1, X2 tramite fattori (0.5, 1.2) con eps ~ N(0,1)
factors <- matrix(c(0.5, 1.2, 0.5, 1.2), ncol = 2)
X_dep <- X_iid[, 1:2] %*% factors + matrix(rnorm(n_train * 2), nrow = n_train, ncol = 2)
# Matrice completa X (6 predittori)
X_train <- cbind(X_iid, X_dep)
colnames(X_train) <- paste0("X", 1:6)
# Risposta: y = bernoulli(p) con p = 1/(1+exp(-eta)), dipende solo da X1, X2, X3
eta <- rowSums(X_train[, 1:3]) + rnorm(n_train, mean = 0, sd = 1)
p_train <- 1 / (1 + exp(-eta)) # logit
y_train <- rbinom(n_train, size = 1, prob = p_train)

train <- data.frame(y = y_train, X_train)


# TEST
# allo stesso modo
X_iid_test <- matrix(rnorm(n_test * 4), nrow = n_test, ncol = 4)
X_dep_test <- X_iid_test[, 1:2] %*% factors + matrix(rnorm(n_test * 2), nrow = n_test, ncol = 2)
X_test <- cbind(X_iid_test, X_dep_test)
colnames(X_test)  <- paste0("X", 1:6)
eta_test <- rowSums(X_test[, 1:3]) + rnorm(n_test, mean = 0, sd = 1)
p_test <- 1 / (1 + exp(-eta_test))
y_test <- rbinom(n_test, size = 1, prob = p_test)

test <- data.frame(y = y_test, X_test)


# analisi correlazioni
cor(test[,-1])
ggplot(melt(cor(test[,-1])), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  coord_equal()



# Bayesian Model Averaging ------------------------------------------------


X <- data.frame(intercept = rep(1, nrow(X_train)), X_train)
models <- as.matrix(expand.grid(1, c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1)))

# Definizione dei link disponibili
links <- c("logit", "probit", "cloglog")   
# nomi dei file Stan: logit.stan, probit.stan, cloglog.stan

# Griglia modelli: combinazioni di covariate (come prima) × link
models_link <- expand.grid(1:nrow(models), links)
colnames(models_link) <- c("model_idx", "link")
models_link <- cbind(models_link, models)

logit_mod   <- stan_model("logit.stan")
probit_mod  <- stan_model("probit.stan")
cloglog_mod <- stan_model("cloglog.stan")

model_list_link <- vector("list", nrow(models_link))
loglik_list <- vector("list", nrow(models_link))
WAIC_vec <- numeric(nrow(models_link))
out_coef_values <- vector("list", nrow(models_link))

pb <- txtProgressBar(min = 1, max = nrow(models_link), style = 3)
for(m in 1:nrow(models_link)){
  setTxtProgressBar(pb, m)
  
  idx <- models_link$model_idx[m]
  link_name <- models_link$link[m]
  
  # drop = FALSE per evitare che R trasformi una singola colonna in un vettore
  X_temp <- as.matrix(X[, as.logical(models[idx,]), drop = FALSE])
  
  data_temp <- list(
    n = nrow(X_temp),
    p = ncol(X_temp),
    y = as.integer(y_train),
    X = X_temp,
    beta0 = array(0, dim = ncol(X_temp)), 
    Sigma0 = diag(10, ncol(X_temp))  
  )
  
  # in teoria forza link_name a essere una stringa di testo
  stan_mod <- switch(as.character(link_name),
                     "logit" = logit_mod,
                     "probit" = probit_mod,
                     "cloglog" = cloglog_mod)
  
  fit <- sampling(stan_mod,
                  data = data_temp,
                  chains = 1,
                  iter = 1500,
                  warmup = 500,
                  seed = 42)
  
  model_list_link[[m]] <- fit
  
  # log-likelihood (NON exp)
  log_lik <- rstan::extract(fit, pars = "log_lik")[[1]]
  loglik_list[[m]] <- log_lik
  
  # WAIC (versione stabile)
  lppd <- sum(log(colMeans(exp(log_lik))))
  p_waic <- sum(apply(log_lik, 2, var))
  WAIC_vec[m] <- -2 * (lppd - p_waic)
  
  # coefficienti
  out_coef_values[[m]] <- rstan::extract(fit, pars = c("beta"))
}
close(pb)

# save(model_list_link, loglik_list, WAIC_vec, out_coef_values, file = "BMA_results_new.RData")
# load("BMA_results_new.RData")

# Calcolo dei pesi BMA (corretto)
logml <- sapply(model_list_link, function(mod) {
  bridge_sampler(mod)$logml
})

wei_link <- exp(logml - max(logml))
wei_link <- wei_link / sum(wei_link)

# per estrarre link 
mm <- numeric(nrow(models_link))
for(m in 1:nrow(models_link)){
  mm[m] <- attr(model_list_link[[m]]@stanmodel@model_code, "model_name2")
}

# tabella risultati BMA
tab <- print(cbind(
  mod = 1:192,
  WAIC_vec = sprintf("%.4f", as.numeric(WAIC_vec)),
  wei_link = sprintf("%.4e", as.numeric(wei_link)),
  mm = mm,
  rbind(models, models, models)
), quote = FALSE)

tab[which.max(wei_link),]


# Grafici -----------------------------------------------------------------


models2 <- rbind(models, models, models)
coef <- vector("list", nrow(models2))
for(m in 1:nrow(models2)){
  idx <- which(models2[m, ] == 1)
  n_iter <- nrow(out_coef_values[[m]]$beta)
  temp_coef <- matrix(0, nrow = n_iter, ncol = ncol(models2))
  temp_coef[, idx] <- out_coef_values[[m]]$beta
  coef[[m]] <- temp_coef
}


plot_list <- list()

for(j in 1:7){
  
  idx_beta <- which(models2[, j] == 1)
  
  pip_beta <- sum(wei_link[models2[, j] == 1])
  p_zero <- 1 - pip_beta
  
  df_list <- lapply(idx_beta, function(m){
    data.frame(
      value = coef[[m]][, j],
      model = m,
      weight = wei_link[m]
    )
  })
  
  df <- do.call(rbind, df_list)
  
  # densità BMA
  grid <- seq(min(df$value), max(df$value), length.out = 500)
  dens_bma <- rep(0, length(grid))
  
  for(m in idx_beta){
    d <- density(coef[[m]][, j], from = min(grid), to = max(grid), n = 500)
    dens_bma <- dens_bma + wei_link[m] * d$y
  }
  
  df_bma <- data.frame(x = grid, y = dens_bma)
  x_max <- max(grid)
  spike_height <- p_zero * max(dens_bma)
  
  p <- ggplot(df, aes(x = value, group = model)) +
    
    geom_density(aes(color = weight), alpha = 0.4) +
    
    scale_color_gradient(
      low = "grey80",
      high = "grey20"
    ) +
    
    geom_line(data = df_bma,
              aes(x = x, y = y),
              inherit.aes = FALSE,
              color = "steelblue",
              linewidth = 1.2) +
    
    geom_segment(aes(x = 0, xend = 0, y = 0, yend = spike_height),
                 color = "red", linewidth = 1.5) +
    
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    
    annotate("text",
             x     = x_max * 0.8,
             y     = max(dens_bma) * 0.95,
             label = paste0("Densità BMA"),
             color = "steelblue",
             size  = 3.5) +
    
    labs(
      title = bquote("Distribuzione campioni MCMC di " * beta[.(j)]),
      x = bquote(beta[.(j)]),
      y = "Density"
    ) +
    
    theme_minimal()
  
  plot_list[[j]] <- p
}

for(i in 1:7){
  print(plot_list[[i]])
}

library(gridExtra)
grid.arrange(grobs = plot_list, ncol = 2)


# Accuracy ----------------------------------------------------------------


XX_test <- as.matrix(data.frame(intercept = rep(1, nrow(X_test)), X_test))
p_model_link <- matrix(0, nrow = nrow(models_link), ncol = nrow(X_test))

for(i in 1:nrow(models_link)){
  
  idx <- models_link$model_idx[i]
  beta_hat <- coef_list[[i]]
  
  XX_test_temp <- XX_test[, as.logical(models[idx, ]), drop = FALSE]
  
  # Ora le dimensioni combaciano 
  eta <- XX_test_temp %*% beta_hat
  
  link_name <- models_link$link[i]
  
  p_model_link[i, ] <- switch(link_name,
                              logit = 1 / (1 + exp(-eta)),
                              probit = pnorm(eta),
                              cloglog = 1 - exp(-exp(-eta))
  )
}

p_hat_bma_link <- as.vector(wei_link %*% p_model_link)
y_pred <- ifelse(p_hat_bma_link > 0.5, 1, 0)
plot(wei_link, col=as.factor(mm)); text(x = seq_along(wei_link), y = wei_link, pos = 2, cex = 0.7, col=as.numeric(as.factor(mm)))


# Calcolo accuratezza finale
mean(y_pred == y_test)


# CONFRONTO COVERAGE: BMA vs BEST, NULLO, FULL, CORRETTO per ogni link -------


# Calcolo le probabilità predittive per tutti i modelli 
p_pred_list_link <- list()
XX_test_t <- t(XX_test) 

for(i in 1:nrow(models_link)){
  idx_var <- models_link$model_idx[i]
  link_name <- as.character(models_link$link[i])
  
  # Estraggo i draw MCMC
  beta_draws <- rstan::extract(model_list_link[[i]], pars = "beta")[[1]]
  
  # Ricostruisco la matrice dei beta completi (7 colonne)
  beta_full <- matrix(0, nrow = nrow(beta_draws), ncol = 7)
  beta_full[, as.logical(models[idx_var, ])] <- beta_draws
  
  # Calcolo predittore lineare ed applico il link corretto
  eta <- beta_full %*% XX_test_t
  p_pred_list_link[[i]] <- switch(link_name,
                                  "logit"   = 1 / (1 + exp(-eta)),
                                  "probit"  = pnorm(eta),
                                  "cloglog" = 1 - exp(-exp(-eta)))
}

# MIstura BMA (Campionamento pesato) 
S_total <- 2000
p_pred_bma_link <- NULL

for(i in 1:nrow(models_link)){
  S_i <- round(wei_link[i] * S_total)
  
  if(S_i > 0){
    draws_i <- p_pred_list_link[[i]]
    idx_samp <- sample(1:nrow(draws_i), S_i, replace = TRUE)
    p_pred_bma_link <- rbind(p_pred_bma_link, draws_i[idx_samp, , drop = FALSE])
  }
}

# Quantili, Width e Coverage per il BMA 
lower_bma <- apply(p_pred_bma_link, 2, quantile, 0.05)
upper_bma <- apply(p_pred_bma_link, 2, quantile, 0.95)
width_bma <- mean(upper_bma - lower_bma)

coverage_BMA <- mean(
  (y_test == 1 & upper_bma >= 0.5) |
    (y_test == 0 & lower_bma <= 0.5)
)

compute_coverage_link <- function(model_index, X_test_matrix, y_test, models_link, models, model_list_link) {
  idx_var <- models_link$model_idx[model_index]
  link_name <- as.character(models_link$link[model_index])
  
  beta_draws <- rstan::extract(model_list_link[[model_index]], pars = "beta")[[1]]
  beta_full <- matrix(0, nrow = nrow(beta_draws), ncol = 7)
  beta_full[, as.logical(models[idx_var, ])] <- beta_draws
  
  eta <- beta_full %*% t(X_test_matrix)
  
  p_pred <- switch(link_name,
                   "logit"   = 1 / (1 + exp(-eta)),
                   "probit"  = pnorm(eta),
                   "cloglog" = 1 - exp(-exp(-eta)))
  
  lower <- apply(p_pred, 2, quantile, 0.05)
  upper <- apply(p_pred, 2, quantile, 0.95)
  
  coverage <- mean((y_test == 1 & upper >= 0.5) | (y_test == 0 & lower <= 0.5))
  width <- mean(upper - lower)
  
  return(list(coverage = coverage, width = width))
}


# trovo l'indice assoluto del modello migliore in assoluto (con peso maggiore)
idx_best_link <- which.max(wei_link)


# questa va tipo a cercare tra i modelli per es. nulli qual è il migliore
get_best_link_idx <- function(target_var_idx) {
  # righe in models_link corrispondenti a questa combinazione di variabili
  candidate_rows <- which(models_link$model_idx == target_var_idx)
  # tra queste 3 trova quella con il peso (wei_link) più alto
  best_row <- candidate_rows[which.max(wei_link[candidate_rows])]
  return(best_row)
}

# indici dei tuoi modelli di riferimento
idx_null    <- get_best_link_idx(1)   # Modello Nullo (solo intercetta)
idx_full    <- get_best_link_idx(64)  # Modello Full (tutte le 6 variabili)
idx_correct <- get_best_link_idx(8)   # Modello Corretto (X1, X2, X3)

# calcolo il coverage per ciascuno usando la funzione di prima
cov_best    <- compute_coverage_link(idx_best_link, XX_test, y_test, 
                                     models_link, models, model_list_link)
cov_null    <- compute_coverage_link(idx_null, XX_test, y_test, models_link, 
                                     models, model_list_link)
cov_full    <- compute_coverage_link(idx_full, XX_test, y_test, models_link, 
                                     models, model_list_link)
cov_correct <- compute_coverage_link(idx_correct, XX_test, y_test, models_link, 
                                     models, model_list_link)

# questa non ha best nullo full per tutti i link ma solo logit
risultati_completi <- data.frame(
  Model = c(
    "BMA Multi-Link",
    paste0("Best Overall (", models_link$link[idx_best_link], ")"),
    paste0("Nullo (", models_link$link[idx_null], ")"),
    paste0("Full (", models_link$link[idx_full], ")"),
    paste0("Corretto (", models_link$link[idx_correct], ")")
  ),
  Coverage = c(
    coverage_BMA, 
    cov_best$coverage, 
    cov_null$coverage, 
    cov_full$coverage, 
    cov_correct$coverage
  ),
  Width = c(
    width_bma, 
    cov_best$width, 
    cov_null$width, 
    cov_full$width, 
    cov_correct$width
  )
)

print(risultati_completi)


# altra funzione
get_exact_model_idx <- function(target_var_idx, target_link) {
  # cerca la riga esatta che corrisponde a quelle variabili E a quel link
  which(models_link$model_idx == target_var_idx & models_link$link == target_link)
}

# combinazioni di variabili che vogliamo analizzare (Nullo=1, Corretto=8, Full=64)
nomi_modelli <- c("Nullo", "Corretto", "Full")
indici_variabili <- c(1, 8, 64)
links_disponibili <- c("logit", "probit", "cloglog")

# lista per salvare i risultati riga per riga
lista_risultati <- list()

# Aggiungo per primi il BMA e il Best Overall (calcolati in precedenza)
lista_risultati[[1]] <- data.frame(Model = "BMA Multi-Link", 
                                   Coverage = coverage_BMA, 
                                   Width = width_bma)
lista_risultati[[2]] <- data.frame(Model = paste0("Best Overall (", 
                                                  models_link$link[idx_best_link], ")"), 
                                   Coverage = cov_best$coverage, 
                                   Width = cov_best$width)

# tutti e 3 i link
contatore <- 3
for (i in 1:length(nomi_modelli)) {
  for (l in links_disponibili) {
    
    # Trovo l'indice di questa esatta combinazione (es. "Nullo" + "probit")
    idx_esatto <- get_exact_model_idx(indici_variabili[i], l)
    
    # Calcolo il coverage
    cov_esatto <- compute_coverage_link(idx_esatto, XX_test, y_test, 
                                        models_link, models, model_list_link)
    
    # Salvo il risultato
    lista_risultati[[contatore]] <- data.frame(
      Model = paste0(nomi_modelli[i], " (", l, ")"),
      Coverage = cov_esatto$coverage,
      Width = cov_esatto$width
    )
    
    contatore <- contatore + 1
  }
}

tabella_totale <- do.call(rbind, lista_risultati)
print(tabella_totale)



# =========================
# WAIC + LPML + ACCURACY + TABELLA (SIMULATI)
# =========================

N_train <- nrow(X_train)
S_total <- 2000

CPO_matrix_sim <- matrix(0, nrow = nrow(models_link), ncol = N_train)
WAIC_models_sim <- numeric(nrow(models_link))
LPML_models_sim <- numeric(nrow(models_link))

pooled_log_lik_bma_sim <- matrix(NA, nrow = S_total, ncol = N_train)
current_row <- 1

pb_sim <- txtProgressBar(min = 1, max = nrow(models_link), style = 3)

for(i in 1:nrow(models_link)){
  setTxtProgressBar(pb_sim, i)
  
  log_lik <- loglik_list[[i]]
  
  # WAIC
  lppd_i <- sum(log(colMeans(exp(log_lik))))
  p_waic_i <- sum(apply(log_lik, 2, var))
  WAIC_models_sim[i] <- -2 * (lppd_i - p_waic_i)
  
  # LPML
  cpo_i <- 1 / colMeans(exp(-log_lik))
  CPO_matrix_sim[i, ] <- cpo_i
  LPML_models_sim[i] <- sum(log(cpo_i))
  
  # pooling per BMA
  S_i <- round(wei_link[i] * S_total)
  
  if(S_i > 0){
    idx_samp <- sample(1:nrow(log_lik), S_i, replace = TRUE)
    pooled_log_lik_bma_sim[current_row:(current_row + S_i - 1), ] <- log_lik[idx_samp, ]
    current_row <- current_row + S_i
  }
}
close(pb_sim)

pooled_log_lik_bma_sim <- na.omit(pooled_log_lik_bma_sim)

# BMA WAIC + LPML
CPO_bma_sim <- as.vector(wei_link %*% CPO_matrix_sim)
LPML_BMA_sim <- sum(log(CPO_bma_sim))

lppd_bma_sim <- sum(log(colMeans(exp(pooled_log_lik_bma_sim))))
p_waic_bma_sim <- sum(apply(pooled_log_lik_bma_sim, 2, var))
WAIC_BMA_sim <- -2 * (lppd_bma_sim - p_waic_bma_sim)


# =========================
# ACCURACY
# =========================

coef_list_sim <- lapply(out_coef_values, function(x) colMeans(x$beta))

p_model_link <- matrix(0, nrow = nrow(models_link), ncol = nrow(X_test))

for(i in 1:nrow(models_link)){
  
  idx <- models_link$model_idx[i]
  beta_hat <- coef_list_sim[[i]]
  
  XX_test_temp <- XX_test[, as.logical(models[idx, ]), drop = FALSE]
  eta <- XX_test_temp %*% beta_hat
  
  link_name <- as.character(models_link$link[i])
  
  p_model_link[i, ] <- switch(link_name,
                              logit = 1 / (1 + exp(-eta)),
                              probit = pnorm(eta),
                              cloglog = 1 - exp(-exp(-eta)))
}

p_hat_bma_link <- as.vector(wei_link %*% p_model_link)
y_pred <- ifelse(p_hat_bma_link > 0.5, 1, 0)

accuracy_BMA_sim <- mean(y_pred == y_test)


# =========================
# FUNZIONI DI SUPPORTO
# =========================

get_metrics_sim <- function(model_index) {
  list(
    waic = WAIC_models_sim[model_index],
    lpml = LPML_models_sim[model_index]
  )
}


# =========================
# COSTRUZIONE TABELLA
# =========================

lista_risultati_sim <- list()

# BMA
lista_risultati_sim[[1]] <- data.frame(
  Model = "BMA Multi-Link",
  Coverage = coverage_BMA,
  Width = width_bma,
  WAIC = WAIC_BMA_sim,
  LPML = LPML_BMA_sim,
  Accuracy = accuracy_BMA_sim
)

# Best overall
idx_best_link <- which.max(wei_link)

cov_best <- compute_coverage_link(idx_best_link, XX_test, y_test, 
                                  models_link, models, model_list_link)

metrics_best <- get_metrics_sim(idx_best_link)

acc_best <- mean((p_model_link[idx_best_link, ] > 0.5) == y_test)

lista_risultati_sim[[2]] <- data.frame(
  Model = paste0("Best Overall (", models_link$link[idx_best_link], ")"),
  Coverage = cov_best$coverage,
  Width = cov_best$width,
  WAIC = metrics_best$waic,
  LPML = metrics_best$lpml,
  Accuracy = acc_best
)

# Modelli specifici
nomi_modelli <- c("Nullo", "Corretto", "Full")
indici_variabili <- c(1, 8, 64)
links_disponibili <- c("logit", "probit", "cloglog")

contatore <- 3

for (i in 1:length(nomi_modelli)) {
  for (l in links_disponibili) {
    
    idx_esatto <- get_exact_model_idx(indici_variabili[i], l)
    
    cov_esatto <- compute_coverage_link(idx_esatto, XX_test, y_test, 
                                        models_link, models, model_list_link)
    
    metrics_esatto <- get_metrics_sim(idx_esatto)
    
    acc_esatto <- mean((p_model_link[idx_esatto, ] > 0.5) == y_test)
    
    lista_risultati_sim[[contatore]] <- data.frame(
      Model = paste0(nomi_modelli[i], " (", l, ")"),
      Coverage = cov_esatto$coverage,
      Width = cov_esatto$width,
      WAIC = metrics_esatto$waic,
      LPML = metrics_esatto$lpml,
      Accuracy = acc_esatto
    )
    
    contatore <- contatore + 1
  }
}

tabella_simulati <- do.call(rbind, lista_risultati_sim)

print(tabella_simulati)


# DATI REALI --------------------------------------------------------------

# usiamo i dati pima sul diabete (ci sono due dataset uno di train e uno di test)
data(Pima.tr)
data(Pima.te)

# variabile risposta (trasformata in 0 e 1)
y_train_pima <- ifelse(Pima.tr$type == "Yes", 1, 0)
y_test_pima <- ifelse(Pima.te$type == "Yes", 1, 0)

# covariate (7 variabili)
X_train_raw <- Pima.tr[, 1:7]
X_test_raw <- Pima.te[, 1:7]

# standardizzazione (non so se necessaria)
X_train_scaled <- scale(X_train_raw)
X_test_scaled  <- scale(X_test_raw, 
                        center = attr(X_train_scaled, "scaled:center"), 
                        scale  = attr(X_train_scaled, "scaled:scale"))

# matrici finali con intercetta
X_pima <- as.matrix(data.frame(intercept = rep(1, nrow(X_train_scaled)), 
                               X_train_scaled))
XX_test_pima <- as.matrix(data.frame(intercept = rep(1, nrow(X_test_scaled)), 
                                     X_test_scaled))



# Bayesian Model Averaging PIMA -------------------------------------------


# 8 colonne (1 intercetta + 7 covariate) -> 128 combinazioni
models_pima <- as.matrix(expand.grid(1, c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1)))

# 128 x 3 = 384 modelli totali
links <- c("logit", "probit", "cloglog")   
models_link_pima <- expand.grid(1:nrow(models_pima), links)
colnames(models_link_pima) <- c("model_idx", "link")

# Compilo i file stan fuori dal ciclo come facevamo prima con dati simulati
#     logit_mod   <- stan_model("logit.stan")
#     probit_mod  <- stan_model("probit.stan")
#     cloglog_mod <- stan_model("cloglog.stan")


model_list_pima <- vector("list", nrow(models_link_pima))
coef_list_pima  <- vector("list", nrow(models_link_pima))
WAIC_vec_pima   <- numeric(nrow(models_link_pima))


pb <- txtProgressBar(min = 1, max = nrow(models_link_pima), style = 3)

for(m in 1:nrow(models_link_pima)){
  setTxtProgressBar(pb, m)
  
  idx <- models_link_pima$model_idx[m]
  link_name <- models_link_pima$link[m]
  
  # Estraggo solo le colonne attive per questo modello
  X_temp <- as.matrix(X_pima[, as.logical(models_pima[idx,]), drop = FALSE])
  
  data_temp <- list(
    n = nrow(X_temp),
    p = ncol(X_temp),
    y = as.integer(y_train_pima),
    X = X_temp,
    beta0 = array(0, dim = ncol(X_temp)), 
    Sigma0 = diag(10, ncol(X_temp))  
  )
  
  stan_mod <- switch(as.character(link_name),
                     "logit"   = logit_mod,
                     "probit"  = probit_mod,
                     "cloglog" = cloglog_mod)
  
  fit <- sampling(stan_mod,
                  data = data_temp,
                  chains = 1,
                  iter = 1500,
                  warmup = 500,
                  seed = 42,
                  refresh = 0) 
  # refresh=0 nasconde i print continui di Stan in console
  
  model_list_pima[[m]] <- fit
  
  # WAIC
  log_lik <- rstan::extract(fit, pars = "log_lik")[[1]]
  lppd <- sum(log(colMeans(exp(log_lik))))
  p_waic <- sum(apply(log_lik, 2, var))
  WAIC_vec_pima[m] <- -2 * (lppd - p_waic)
  
  # Coefficienti medi
  coef_list_pima[[m]] <- colMeans(rstan::extract(fit, "beta")[[1]])
}
close(pb)

# save(model_list_pima, WAIC_vec_pima, WAIC_vec_pima, coef_list_pima, file = "BMA_results_pima.RData")
# load("BMA_results_pima.RData")


# calcolo pesi con bridge sampling
logml_pima <- sapply(model_list_pima, function(mod) bridge_sampler(mod, silent = TRUE)$logml)
wei_link_pima <- exp(logml_pima - max(logml_pima))
wei_link_pima <- wei_link_pima / sum(wei_link_pima)

# accuracy del BMA
p_model_pima <- matrix(0, nrow = nrow(models_link_pima), ncol = nrow(XX_test_pima))


# Accuracy PIMA -----------------------------------------------------------


for(i in 1:nrow(models_link_pima)){
  idx <- models_link_pima$model_idx[i]
  beta_hat <- coef_list_pima[[i]]
  
  XX_test_temp <- XX_test_pima[, as.logical(models_pima[idx, ]), drop = FALSE]
  eta <- XX_test_temp %*% beta_hat
  link_name <- as.character(models_link_pima$link[i])
  
  p_model_pima[i, ] <- switch(link_name,
                              "logit"   = 1 / (1 + exp(-eta)),
                              "probit"  = pnorm(eta),
                              "cloglog" = 1 - exp(-exp(-eta)))
}

p_hat_bma_pima <- as.vector(wei_link_pima %*% p_model_pima)
y_pred_pima <- ifelse(p_hat_bma_pima > 0.5, 1, 0)

acc_BMA_pima <- mean(y_pred_pima == y_test_pima)
cat("\nAccuratezza BMA su Test Set:", acc_BMA_pima, "\n")



# Coverage PIMA -----------------------------------------------------------



p_pred_list_pima <- list()
XX_test_t_pima <- t(XX_test_pima) 

for(i in 1:nrow(models_link_pima)){
  idx_var <- models_link_pima$model_idx[i]
  link_name <- as.character(models_link_pima$link[i])
  
  beta_draws <- rstan::extract(model_list_pima[[i]], pars = "beta")[[1]]
  
  beta_full <- matrix(0, nrow = nrow(beta_draws), ncol = 8)
  beta_full[, as.logical(models_pima[idx_var, ])] <- beta_draws
  
  eta <- beta_full %*% XX_test_t_pima
  
  p_pred_list_pima[[i]] <- switch(link_name,
                                  "logit"   = 1 / (1 + exp(-eta)),
                                  "probit"  = pnorm(eta),
                                  "cloglog" = 1 - exp(-exp(-eta))
  )
}

S_total <- 2000
p_pred_bma_pima <- NULL

for(i in 1:nrow(models_link_pima)){
  S_i <- round(wei_link_pima[i] * S_total)
  
  if(S_i > 0){
    draws_i <- p_pred_list_pima[[i]]
    idx_samp <- sample(1:nrow(draws_i), S_i, replace = TRUE)
    p_pred_bma_pima <- rbind(p_pred_bma_pima, draws_i[idx_samp, , drop = FALSE])
  }
}

lower_bma <- apply(p_pred_bma_pima, 2, quantile, 0.05)
upper_bma <- apply(p_pred_bma_pima, 2, quantile, 0.95)

width_bma_pima <- mean(upper_bma - lower_bma)

coverage_BMA_pima <- mean(
  (y_test_pima == 1 & upper_bma >= 0.5) |
    (y_test_pima == 0 & lower_bma <= 0.5)
)


compute_coverage_pima <- function(model_index) {
  
  idx_var <- models_link_pima$model_idx[model_index]
  link_name <- as.character(models_link_pima$link[model_index])
  
  beta_draws <- rstan::extract(model_list_pima[[model_index]], pars = "beta")[[1]]
  
  beta_full <- matrix(0, nrow = nrow(beta_draws), ncol = 8)
  beta_full[, as.logical(models_pima[idx_var, ])] <- beta_draws
  
  eta <- beta_full %*% t(XX_test_pima)
  
  p_pred <- switch(link_name,
                   "logit"   = 1 / (1 + exp(-eta)),
                   "probit"  = pnorm(eta),
                   "cloglog" = 1 - exp(-exp(-eta))
  )
  
  lower <- apply(p_pred, 2, quantile, 0.05)
  upper <- apply(p_pred, 2, quantile, 0.95)
  
  coverage <- mean((y_test_pima == 1 & upper >= 0.5) |
                     (y_test_pima == 0 & lower <= 0.5))
  
  width <- mean(upper - lower)
  
  return(list(coverage = coverage, width = width))
}

# indici modelli
get_best_link_idx_pima <- function(target_var_idx) {
  candidate_rows <- which(models_link_pima$model_idx == target_var_idx)
  best_row <- candidate_rows[which.max(wei_link_pima[candidate_rows])]
  return(best_row)
}

get_exact_model_idx_pima <- function(target_var_idx, target_link) {
  which(models_link_pima$model_idx == target_var_idx &
          models_link_pima$link == target_link)
}

idx_best_pima <- which.max(wei_link_pima)

nomi_modelli_pima <- c("Nullo", "Full")
indici_variabili_pima <- c(1, 128)
links_disponibili <- c("logit", "probit", "cloglog")

lista_risultati_pima <- list()

# BMA
lista_risultati_pima[[1]] <- data.frame(
  Model = "BMA Multi-Link",
  Coverage = coverage_BMA_pima,
  Width = width_bma_pima
)

# Best overall
cov_best_pima <- compute_coverage_pima(idx_best_pima)

lista_risultati_pima[[2]] <- data.frame(
  Model = paste0("Best Overall (", models_link_pima$link[idx_best_pima], ")"),
  Coverage = cov_best_pima$coverage,
  Width = cov_best_pima$width
)

contatore <- 3

for (i in 1:length(nomi_modelli_pima)) {
  for (l in links_disponibili) {
    
    idx_esatto <- get_exact_model_idx_pima(indici_variabili_pima[i], l)
    
    cov_esatto <- compute_coverage_pima(idx_esatto)
    
    lista_risultati_pima[[contatore]] <- data.frame(
      Model = paste0(nomi_modelli_pima[i], " (", l, ")"),
      Coverage = cov_esatto$coverage,
      Width = cov_esatto$width
    )
    
    contatore <- contatore + 1
  }
}

tabella_totale_pima <- do.call(rbind, lista_risultati_pima)

print(tabella_totale_pima)


# --- Calcolo WAIC e LPML per tutti i modelli e preparazione dati BMA ---

N_train <- nrow(X_pima)
CPO_matrix <- matrix(0, nrow = nrow(models_link_pima), ncol = N_train)
WAIC_models <- numeric(nrow(models_link_pima))
LPML_models <- numeric(nrow(models_link_pima))

# Prepariamo la matrice per unire le log-lik per il BMA
pooled_log_lik_bma <- matrix(NA, nrow = S_total, ncol = N_train)
current_row <- 1

cat("\nEstrazione log_lik e calcolo WAIC/LPML (train set) in corso...\n")
pb_met <- txtProgressBar(min = 1, max = nrow(models_link_pima), style = 3)

for(i in 1:nrow(models_link_pima)){
  setTxtProgressBar(pb_met, i)
  
  # Estraiamo la log-verosimiglianza dal fit salvato
  log_lik <- rstan::extract(model_list_pima[[i]], pars = "log_lik")[[1]]
  
  # --- Calcolo metriche per il singolo modello (i) ---
  
  # WAIC
  lppd_i <- sum(log(colMeans(exp(log_lik))))
  p_waic_i <- sum(apply(log_lik, 2, var))
  WAIC_models[i] <- -2 * (lppd_i - p_waic_i)
  
  # LPML (usando l'inverso della likelihood per calcolare le CPO)
  # Media armonica delle likelihoods a posteriori
  cpo_i <- 1 / colMeans(exp(-log_lik)) 
  CPO_matrix[i, ] <- cpo_i
  LPML_models[i] <- sum(log(cpo_i))
  
  # --- Costruzione della matrice log_lik miscelata per il BMA ---
  
  S_i <- round(wei_link_pima[i] * S_total)
  if(S_i > 0){
    # Campioniamo casualmente S_i righe dalla log_lik di questo modello
    idx_samp <- sample(1:nrow(log_lik), S_i, replace = TRUE)
    
    # Riempiamo la porzione corrispondente nella matrice combinata
    pooled_log_lik_bma[current_row:(current_row + S_i - 1), ] <- log_lik[idx_samp, ]
    current_row <- current_row + S_i
  }
}
close(pb_met)

# Pulizia di eventuali NA in fondo alla matrice (se arrotondamenti != S_total)
pooled_log_lik_bma <- na.omit(pooled_log_lik_bma)

# LPML BMA: La somma dei logaritmi delle CPO ponderate
CPO_bma <- as.vector(wei_link_pima %*% CPO_matrix)
LPML_BMA_val <- sum(log(CPO_bma))

# WAIC BMA: Calcolato sulla matrice combinata
lppd_bma <- sum(log(colMeans(exp(pooled_log_lik_bma))))
p_waic_bma <- sum(apply(pooled_log_lik_bma, 2, var))
WAIC_BMA_val <- -2 * (lppd_bma - p_waic_bma)

# Helper per recuperare WAIC e LPML dal singolo modello
get_metrics_pima <- function(model_index) {
  list(waic = WAIC_models[model_index], lpml = LPML_models[model_index])
}

lista_risultati_pima <- list()

# BMA
lista_risultati_pima[[1]] <- data.frame(
  Model = "BMA Multi-Link",
  Coverage = coverage_BMA_pima,
  Width = width_bma_pima,
  WAIC = WAIC_BMA_val,
  LPML = LPML_BMA_val
)

# Best overall
cov_best_pima <- compute_coverage_pima(idx_best_pima)
metrics_best_pima <- get_metrics_pima(idx_best_pima)

lista_risultati_pima[[2]] <- data.frame(
  Model = paste0("Best Overall (", models_link_pima$link[idx_best_pima], ")"),
  Coverage = cov_best_pima$coverage,
  Width = cov_best_pima$width,
  WAIC = metrics_best_pima$waic,
  LPML = metrics_best_pima$lpml
)

contatore <- 3

for (i in 1:length(nomi_modelli_pima)) {
  for (l in links_disponibili) {
    
    idx_esatto <- get_exact_model_idx_pima(indici_variabili_pima[i], l)
    
    cov_esatto <- compute_coverage_pima(idx_esatto)
    metrics_esatto <- get_metrics_pima(idx_esatto)
    
    lista_risultati_pima[[contatore]] <- data.frame(
      Model = paste0(nomi_modelli_pima[i], " (", l, ")"),
      Coverage = cov_esatto$coverage,
      Width = cov_esatto$width,
      WAIC = metrics_esatto$waic,
      LPML = metrics_esatto$lpml
    )
    
    contatore <- contatore + 1
  }
}

tabella_totale_pima <- do.call(rbind, lista_risultati_pima)

# Stampa i risultati!
print(tabella_totale_pima)


# NUOVO CON ACCURACY ------------------------------------------------------
# --- Calcolo WAIC e LPML per tutti i modelli e preparazione dati BMA ---
set.seed(123)
N_train <- nrow(X_pima)
CPO_matrix <- matrix(0, nrow = nrow(models_link_pima), ncol = N_train)
WAIC_models <- numeric(nrow(models_link_pima))
LPML_models <- numeric(nrow(models_link_pima))

# Prepariamo la matrice per unire le log-lik per il BMA
pooled_log_lik_bma <- matrix(NA, nrow = S_total, ncol = N_train)
current_row <- 1

cat("\nEstrazione log_lik e calcolo WAIC/LPML (train set) in corso...\n")
pb_met <- txtProgressBar(min = 1, max = nrow(models_link_pima), style = 3)

for(i in 1:nrow(models_link_pima)){
  setTxtProgressBar(pb_met, i)
  
  log_lik <- rstan::extract(model_list_pima[[i]], pars = "log_lik")[[1]]
  
  # WAIC
  lppd_i <- sum(log(colMeans(exp(log_lik))))
  p_waic_i <- sum(apply(log_lik, 2, var))
  WAIC_models[i] <- -2 * (lppd_i - p_waic_i)
  
  # LPML
  cpo_i <- 1 / colMeans(exp(-log_lik)) 
  CPO_matrix[i, ] <- cpo_i
  LPML_models[i] <- sum(log(cpo_i))
  
  # pooling BMA
  S_i <- round(wei_link_pima[i] * S_total)
  if(S_i > 0){
    idx_samp <- sample(1:nrow(log_lik), S_i, replace = TRUE)
    pooled_log_lik_bma[current_row:(current_row + S_i - 1), ] <- log_lik[idx_samp, ]
    current_row <- current_row + S_i
  }
}
close(pb_met)

pooled_log_lik_bma <- na.omit(pooled_log_lik_bma)

# LPML BMA
CPO_bma <- as.vector(wei_link_pima %*% CPO_matrix)
LPML_BMA_val <- sum(log(CPO_bma))

# WAIC BMA
lppd_bma <- sum(log(colMeans(exp(pooled_log_lik_bma))))
p_waic_bma <- sum(apply(pooled_log_lik_bma, 2, var))
WAIC_BMA_val <- -2 * (lppd_bma - p_waic_bma)


# =========================
# ACCURACY (AGGIUNTA QUI)
# =========================

p_model_pima <- matrix(0, nrow = nrow(models_link_pima), ncol = nrow(XX_test_pima))

for(i in 1:nrow(models_link_pima)){
  
  idx <- models_link_pima$model_idx[i]
  beta_hat <- coef_list_pima[[i]]
  
  XX_test_temp <- XX_test_pima[, as.logical(models_pima[idx, ]), drop = FALSE]
  eta <- XX_test_temp %*% beta_hat
  
  link_name <- as.character(models_link_pima$link[i])
  
  p_model_pima[i, ] <- switch(link_name,
                              "logit"   = 1 / (1 + exp(-eta)),
                              "probit"  = pnorm(eta),
                              "cloglog" = 1 - exp(-exp(-eta)))
}

p_hat_bma_pima <- as.vector(wei_link_pima %*% p_model_pima)
y_pred_pima <- ifelse(p_hat_bma_pima > 0.5, 1, 0)
accuracy_BMA_pima <- mean(y_pred_pima == y_test_pima)


# Helper
get_metrics_pima <- function(model_index) {
  list(waic = WAIC_models[model_index], lpml = LPML_models[model_index])
}

lista_risultati_pima <- list()

# BMA
lista_risultati_pima[[1]] <- data.frame(
  Model = "BMA Multi-Link",
  Coverage = coverage_BMA_pima,
  Width = width_bma_pima,
  WAIC = WAIC_BMA_val,
  LPML = LPML_BMA_val,
  Accuracy = accuracy_BMA_pima
)

# Best overall
cov_best_pima <- compute_coverage_pima(idx_best_pima)
metrics_best_pima <- get_metrics_pima(idx_best_pima)

acc_best_pima <- mean((p_model_pima[idx_best_pima, ] > 0.5) == y_test_pima)

lista_risultati_pima[[2]] <- data.frame(
  Model = paste0("Best Overall (", models_link_pima$link[idx_best_pima], ")"),
  Coverage = cov_best_pima$coverage,
  Width = cov_best_pima$width,
  WAIC = metrics_best_pima$waic,
  LPML = metrics_best_pima$lpml,
  Accuracy = acc_best_pima
)

contatore <- 3

for (i in 1:length(nomi_modelli_pima)) {
  for (l in links_disponibili) {
    
    idx_esatto <- get_exact_model_idx_pima(indici_variabili_pima[i], l)
    
    cov_esatto <- compute_coverage_pima(idx_esatto)
    metrics_esatto <- get_metrics_pima(idx_esatto)
    
    acc_esatto <- mean((p_model_pima[idx_esatto, ] > 0.5) == y_test_pima)
    
    lista_risultati_pima[[contatore]] <- data.frame(
      Model = paste0(nomi_modelli_pima[i], " (", l, ")"),
      Coverage = cov_esatto$coverage,
      Width = cov_esatto$width,
      WAIC = metrics_esatto$waic,
      LPML = metrics_esatto$lpml,
      Accuracy = acc_esatto
    )
    
    contatore <- contatore + 1
  }
}

tabella_totale_pima <- do.call(rbind, lista_risultati_pima)

print(tabella_totale_pima)

# BAGGING -----------------------------------------------------------------

# mtry = 7 così faccio bagging (library per random forest)
set.seed(123) 
modello_bagging <- randomForest(type ~ ., 
                                data = Pima.tr, 
                                mtry = 7,        
                                ntree = 500,    
                                importance = TRUE)

previsioni_bag <- predict(modello_bagging, newdata = Pima.te)

# accuracy
accuracy_bagging <- mean(previsioni_bag == Pima.te$type)
cat("Accuracy Bagging sul Test Set:", accuracy_bagging, "\n")


# variable importance
varImpPlot(modello_bagging, main = "Variable importance",
           col = "darkblue", pch = 16)

gini <- importance(modello_bagging)[, "MeanDecreaseGini"]

gini_rank <- data.frame(
  variable = names(gini),
  MeanDecreaseGini = gini
)

gini_rank <- gini_rank[order(-gini_rank$MeanDecreaseGini), ]
gini_rank

library(ggplot2)

p1 <- ggplot(gini_rank, aes(x = reorder(variable, MeanDecreaseGini), 
                            y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  geom_col() +
  coord_flip() +
  labs(
    x = "Variable",
    y = "Mean Decrease in Gini",
    title = "Variable Importance (Bagging)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  ) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme(legend.position = "none")


# confronto con AUC --------------------------------------------------------


prob_bag <- predict(modello_bagging, newdata = Pima.te, type = "prob")[,2]
auc_bag <- auc(y_test_pima, prob_bag)
cat("AUC Bagging:", auc_bag, "\n")

auc_bma <- auc(y_test_pima, p_hat_bma_pima)
cat("AUC BMA:", auc_bma, "\n")

par(pty = 's')
plot(roc(y_test_pima, prob_bag), col = "blue", main = "ROC Curve")
lines(roc(y_test_pima, p_hat_bma_pima), col = "red")
legend("bottomright", legend = c("Bagging", "BMA"),
       col = c("blue", "red"), lwd = 2)


# variable importance -----------------------------------------------------
# nomi reali delle variabili PIMA
var_names <- colnames(Pima.tr[, 1:7])

pip <- rep(0, length(var_names))

for(j in 1:length(var_names)){
  pip[j] <- sum(wei_link_pima[models_pima[, j+1] == 1]) 
  # +1 perché la prima colonna è intercetta
}

ranking <- data.frame(
  variable = var_names,
  PIP = pip
)

ranking <- ranking[order(-ranking$PIP), ]
ranking

library(viridis)
p2 <- ggplot(ranking, aes(x = reorder(variable, PIP), y = PIP, fill = PIP)) +
  geom_col() +
  coord_flip() +
  labs(
    x = "Variable",
    y = "Posterior Inclusion Probability",
    title = "PIP (BMA)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  ) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 2)

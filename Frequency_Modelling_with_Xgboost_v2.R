###############################
# Modelling with XGBOOST in R #
###############################

# Usefull links
# http://uc-r.github.io/gbm_regression
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# Kaggle study
# https://www.kaggle.com/code/floser/glm-neural-nets-and-xgboost-for-insurance-pricing#2.5-Other-Approaches-and-Summary
# https://www.actuaries.org.uk/news-and-insights/news/article-fitting-data-xgboost

# Libraries
#suppressMessages(library(xgboost))
suppressMessages(library(dplyr))
library(rlang)
#install.packages("rlang")
# https://stackoverflow.com/questions/66782751/namespace-rlang-0-4-5-is-being-loaded-but-0-4-10-is-required
# install.packages("https://cran.r-project.org/src/contrib/Archive/rlang/rlang_1.1.2.tar.gz", repos = NULL, type="source")
# install.packages("Rtools")

library(caret)
#library(ggpubr)
library(tidyr)
library(broom) # convert statistical object into tidy table
library(rpart)
# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects

# Load the data
df = read.csv("C:\\Users\\William\\Documents\\Data Science - ML\\Pricing Project_GLM_vs_GBM\\data.csv")

####################
# Claims Frequency #
####################

# Split train / test
# index <- createDataPartition(df$ClaimNb, p = 0.7, list = FALSE)
# head(index)
# 
# train <- df[index,]
# test <- df[-index,]

set.seed(564738291)
u <- runif(dim(df)[1], min = 0, max = 1)
df$train <- u < 0.7
df$test <- !(df$train)

####################################
# Binning the continuous variables #
####################################
# Work on data --> all as one-hot encoded

glimpse(df)

# CarAge
# We recreate an age band by quantile.
ageband_quantile<-summary(df$CarAge)
print(ageband_quantile)

df$CarAge2 <- cut(df$CarAge, breaks = c(0,3,7,12,Inf),
                  labels = c("0-3","3-7","7-12", ">12"),include.lowest = TRUE)
df %>% group_by(CarAge2) %>% summarise(count=n())

# Drive age
ageband_quantile2<-summary(df$DriverAge)
print(ageband_quantile2)
df$DriverAge2 <- cut(df$DriverAge, breaks = c(18,25,34,44,54,Inf),
                     labels = c("18-25","25-34","34-44", "44-54",">54"),include.lowest = TRUE)
df %>% group_by(DriverAge2) %>% summarise(count=n())

# Density
density_quantile<-summary(df$Density)
print(density_quantile)
df$Density2 <- cut(df$Density, breaks = c(2,67,287,1410,Inf),
                   labels = c("2-67","67-287","287-1410",">1410"),include.lowest = TRUE)
df %>% group_by(Density2) %>% summarise(count=n())

# Applies vtreat to one-hot encode the training and testing data sets
library(vtreat)

# variable names
# Isolate the outcome from the potential predictors and the Exposure
df_train <- df%>% filter(train == TRUE) %>% select(ClaimNb, Exposure,
                                                   Power, CarAge2, DriverAge2, Brand, Gas, Region, Density2)
df_test <- df%>% filter(test == TRUE) %>% select(ClaimNb, Exposure,
                                                 Power, CarAge2, DriverAge2, Brand, Gas, Region, Density2)
features <- setdiff(names(df_train), c("ClaimNb", "Exposure"))
print(features)

# Create the treatment plan from the training data selecting only the features to be used for the training
treatplan <- vtreat::designTreatmentsZ(df_train, features, verbose = FALSE)
print(treatplan)

# Get the "clean" variable names from the scoreFrame
new_vars <- treatplan %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)  
print(new_vars)


# suppressMessages(library(xgboost))
# # construct xgb.DMatrices (these are the internal data structures used by XGBoost during training)  
# dtrain <- xgb.DMatrix(data = data.matrix(X_learn), label = learn$ClaimNb/learn$Exposure, weight=learn$Exposure)
# # make label wrong (shouldn't change anything): change to fold-number
# dtest <- xgb.DMatrix(data = data.matrix(X_test), label = test$fold/test$Exposure, weight=test$Exposure)

# Prepare the training data
features_train <- vtreat::prepare(treatplan, df_train, varRestriction = new_vars) %>% as.matrix()
# type_of(features_train)
# print(features_train)
# colnames(features_train)
response_train <- df_train$ClaimNb/df_train$Exposure # Need the frequency here
#response_train2 <- df_train$ClaimNb


# Prepare the test data
features_test <- vtreat::prepare(treatplan, df_test, varRestriction = new_vars) %>% as.matrix()
response_test <- df_test$ClaimNb/df_test$Exposure
#response_test2 <- df_test$ClaimNb

# dimensions of one-hot encoded data
dim(features_train)
dim(features_test)

# reproducibility
set.seed(123)

##########
# Tuning #
##########

# To tune the XGBoost model we pass parameters as a list object to the params argument. The most common parameters include:

# Parameters:
# eta:controls the learning rate
# max_depth: tree depth
# min_child_weight: minimum number of observations required in each terminal node
# subsample: percent of training data to sample for each tree
# colsample_bytrees: percent of columns to sample from for each tree

# create parameter list
params_list <- list(
  eta = .3,
  max_depth = 5,
  min_child_weight = 2,
  subsample = .8,
  colsample_bytree = .9
)

library(xgboost)
# First model
# Here it incorporates the cross-validation
system.time(xgb.fit.freq <- xgb.cv(
  data = features_train,
  label = response_train, #number claims / exposure
  params = params_list,
  nrounds = 500,# number of trees
  nfold = 5,
  booster = 'gbtree',
  objective = "count:poisson",  # for countinf model
  #eval-metric = "poisson-nloglik", # Negative Log Likelohood, as evaluation metric by default
  tree_method = "hist",
  verbose = 0               # silent,
))

# Check results
head(xgb.fit.freq$evaluation_log)

# get number of trees that minimize error
xgb.fit.freq$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_poisson_nloglik_mean == min(train_poisson_nloglik_mean))[1],
    log_lik.train   = min(train_poisson_nloglik_mean),
    ntrees.test  = which(test_poisson_nloglik_mean == min(test_poisson_nloglik_mean))[1],
    log_lik.test   = min(test_poisson_nloglik_mean),
  )
# ntrees.train log_lik.train ntrees.test log_lik.test
# 1          500     0.3665574          69    0.5270196

# plot error vs number trees
ggplot(xgb.fit.freq$evaluation_log) +
  geom_line(aes(iter, train_poisson_nloglik_mean), color = "red") +
  geom_line(aes(iter, test_poisson_nloglik_mean), color = "blue") + ggtitle("Neg LogLikelihood Evo")


# The test error increases which is an indicator of overfitting,
# meaning that the model is learning the training data too well,
# including its noise and nuances, to the point that it performs poorly on new,
# unseen data (AI google)
#Increases because the model doesn't generalize well to new data, 
#struggling with patterns not present in the training se

#xgb.fit.freq[4]

# xgb.fit.freq$folds[[1]]$evaluation_log


# try another set of parameters

# create parameter list
params_list2 <- list(
  eta = .1,
  max_depth = 1, # choice a stump tree
  min_child_weight = 2,
  subsample = .8,
  colsample_bytree = .9
)

# First model
# Here it incorporates the cross-validation
system.time(xgb.fit.freq2 <- xgb.cv(
  data = features_train,
  label = response_train, #number claims / exposure
  params = params_list2,
  booster = 'gbtree',
  nrounds = 500,# number of trees
  nfold = 5,
  objective = "count:poisson",  # for countinf model
  tree_method = "hist",
  verbose = 0,             # silent,
  #early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
))

# get number of trees that minimize error
xgb.fit.freq2$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_poisson_nloglik_mean == min(train_poisson_nloglik_mean))[1],
    log_lik.train   = min(train_poisson_nloglik_mean),
    ntrees.test  = which(test_poisson_nloglik_mean == min(test_poisson_nloglik_mean))[1],
    log_lik.test   = min(test_poisson_nloglik_mean),
  )

# plot error vs number trees
ggplot(xgb.fit.freq2$evaluation_log) +
  geom_line(aes(iter, train_poisson_nloglik_mean), color = "red") +
  geom_line(aes(iter, test_poisson_nloglik_mean), color = "blue") +
  ggtitle("Neg LogLikelihood Evolution")


xgb.fit.freq2[4]


#####################
# Run a grid search #
#####################

# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1),
  max_depth = c(1, 3, 5),
  min_child_weight = c(1, 3, 5),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_log_lik = 0                     # a place to dump results
)

nrow(hyper_grid)


# grid search 
system.time(for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(123)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = features_train,
    label = response_train,
    nrounds = 500,
    nfold = 5,
    objective = "count:poisson",  # for regression models
    tree_method = "hist",
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_poisson_nloglik_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_poisson_nloglik_mean)
})

# Results
hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)


#########################
# Run the optimal model #
#########################

# Use the xgb.DMatrix to format the data

# Remove the ClaimNb from the data
df_train2 <- subset(df_train, select = -c(ClaimNb, Exposure))
glimpse(df_train2)
glimpse(df_train)

# Formating the data
dtrain <- xgb.DMatrix(data = data.matrix(df_train2), label = df_train$ClaimNb/df_train$Exposure, weight=df_train$Exposure)
glimpse(dtrain)

# create parameter list
params_list3 <- list(
  eta = .1,
  max_depth = 1, # choice a stump tree
  min_child_weight = 1,
  subsample = .8,
  colsample_bytree = .9
)


# Fit the model with the obtained parameters
system.time(xgb.fit.freq.bis <- xgboost(
  data = features_train,
  label = response_train, #number claims / exposure
  params = params_list3,
  nrounds = 129,# number of trees
  objective = "count:poisson",  # for countinf model
  tree_method = "hist",
  verbose = 0               # silent,
))


# create importance matrix
importance_matrix <- xgb.importance(model = xgb.fit.freq.bis)
print(importance_matrix)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")
#?xgb.plot.importance


# Save the model
xgb.save(xgb.fit.freq.bis, "gbm_Claims_freq")

# Load the model
bst <- xgb.load("xgbm_Claims_freq")


# predict values for test data
pred <- predict(bst, features_test)
pred

# results
caret::RMSE(pred, response_test)
## [1] 21319.3



#install.packages("DiagrammeR")
library(DiagrammeR)
xgb.plot.tree(feature_names = colnames(features_train),
              model = xgb.fit.freq.bis,
              trees = 0)

?interaction.plot
interaction.plot(train.comb.cxl$MileBand,train.comb.cxl$TermBand,train.comb.cxl$ClaimCnt,sum)

# Partial Dependance Plot (PDP)

# Better to use them with Continuous variable
pdp <- xgb.fit.freq.bis %>%
  partial(pred.var = "Power_lev_x_d", n.trees = 500, grid.resolution = 100, train = features_train) %>%
  autoplot(rug = TRUE, train = features_train) +
  scale_y_continuous(labels = scales::percent) +
  ggtitle("PDP")

print(pdp)

ice <- xgb.fit.freq.bis %>%
  partial(pred.var = "DriverAge", n.trees = 500, grid.resolution = 100, train = features_train, ice = TRUE) %>%
  autoplot(rug = TRUE, train = features_train, alpha = .1, center = TRUE) +
  scale_y_continuous(labels = scales::dollar) +
  ggtitle("ICE")

plot(ice)

gridExtra::grid.arrange(pdp, ice, nrow = 1)





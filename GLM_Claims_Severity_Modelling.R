###########################################################
# Modelling - Part 1.b - GLM - Claims Severity estimation #
###########################################################

# Libraries
library(dplyr)
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
df<-read.csv("C:\\Users\\William\\Documents\\Data Science - ML\\Pricing Project_GLM_vs_GBM\\data.csv")

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

#########################################
# Training a model for claims severity  #
#########################################

# Unconstrained

# Replace by 0 if NA
df <- df %>% mutate(ClaimAmount = ifelse(is.na(ClaimAmount), 0, ClaimAmount))

#glimpse(df)

#### Create factor for the predictors 
df$Power <- factor(df$Power)
Power.expo = aggregate(df$Exposure, list(df$Power), sum)
df$Power = relevel(df$Power,ref=as.character(Power.expo$Group.1[which(Power.expo$x==max(Power.expo$x))]))

df$Brand <- factor(df$Brand)
Brand.expo = aggregate(df$Exposure, list(df$Brand), sum)
df$Brand = relevel(df$Brand,ref=as.character(Brand.expo$Group.1[which(Brand.expo$x==max(Brand.expo$x))]))

df$Gas <- factor(df$Gas)
Gas.expo = aggregate(df$Exposure, list(df$Gas), sum)
df$Gas = relevel(df$Gas,ref=as.character(Gas.expo$Group.1[which(Gas.expo$x==max(Gas.expo$x))]))

df$Region <- factor(df$Region)
Region.expo = aggregate(df$Exposure, list(df$Region), sum)
df$Region = relevel(df$Region,ref=as.character(Region.expo$Group.1[which(Region.expo$x==max(Region.expo$x))]))

# Step 2: Evaluation of potential predictors
# Test of the different potential covariate
# Set up a grid search
# Set up a grid search
result_grid <- expand.grid(
  covariates = c(1, 'Power', 'CarAge', 'DriverAge', 'Brand', 'Gas', 'Region', 'Density'),
  AIC = NA,
  Deviance = NA)
print(result_grid)

# Run a for loop adding building each time a model with one parameter
for(i in seq_len(nrow(result_grid))) {
  fmla <- as.formula(paste("ClaimAmount ~ ", result_grid$covariates[i]))
  f <- glm(fmla,
           data = df,
           subset = train & ClaimNb > 0,
           family = Gamma(link = "log"),
           offset = log(ClaimNb))
  #rms[v] <- RMSEP(dta$clm.count[dta$train],
  #predict(f, newdata = dta[dta$train,],
  #type = "response"))
  # print(fmla)
  result_grid$AIC[i] <- f$aic
  result_grid$Deviance[i] <- f$deviance
}
knitr::kable(result_grid, format = "markdown")
#clipr::write_clip(result_grid)

# Graph AIC & Deviance
scatter <- ggplot(result_grid, aes(x=AIC, y=Deviance)) +
  geom_point() + # Show dots
  geom_text(
    label=result_grid$covariates, 
    nudge_x = 0.25, nudge_y = 0.25, 
    check_overlap = T
  ) + ggtitle("All single variable model - Claims Severity")
print(scatter)



# Building another model with all the covariates
model_full <- glm(ClaimAmount ~ DriverAge + Region + Density + Power + CarAge + Brand + Gas, 
                  data = df,
                  subset = train & ClaimNb > 0,
                  family = Gamma(link = "log"),
                  weights = ClaimNb,
                  offset = log(ClaimNb))
summary(model_full)

#backward stepwise regression
model_full2 <- step(model_full, direction = "backward")
# Here the brand is not statistically significant.

summary(model_full2)

model_full3 <- glm(ClaimAmount ~ DriverAge + Region + Density + Power + CarAge + Brand + Gas,
                   data = df,
                   subset = train & ClaimNb > 0,
                   family = Gamma(link = "log"))
summary(model_full3)

# Export for table
# tidy_glm_sev <- tidy(model_full )
# clipr::write_clip(tidy_glm_sev)

# Saving the model
saveRDS(model_full3, file = "models/GLM_clm_sev.rda")
#saveRDS(m3, file = "models/GLM_clm_freq_v1.rda")


#loading the model
#model_old = readRDS("models/GLM_clm_freq.rda")

#checking whether the model has been loaded with different name
# ls()


# Prediction on the model
test <- df %>% filter(test == TRUE)

pred <- predict(model_full3, test, type = "response")
head(pred)



  
  
  
  


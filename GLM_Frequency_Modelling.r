############################################################
# Modelling - Part 1.a - GLM - Claims Frequency estimation #
############################################################

# Main Library
library(dplyr)
library(ggplot2)
library(lmtest) # for the Likelihood ratio test

# Load the data
df<-read.csv("C:\\Users\\William\\Documents\\Data Science - ML\\Pricing Project_GLM_vs_GBM\\data.csv")

### Model 1 - All variables

#### Transformation of categorical into factors.

# Transform into factor
df$Power = factor(df$Power)
df$Brand = factor(df$Brand)
df$Region = factor(df$Region)
df$Gas = factor(df$Gas)

# Set reference level for the data
Power.expo=aggregate(df$Exposure, list(df$Power),sum)
Brand.expo=aggregate(df$Exposure, list(df$Brand),sum)
Region.expo=aggregate(df$Exposure, list(df$Region),sum)
Gas.expo=aggregate(df$Exposure, list(df$Gas),sum)

#step 2: set the reference level
df$Power = relevel(df$Power, ref=as.character(Power.expo$Group.1[which(Power.expo$x==max(Power.expo$x))]))
df$Brand = relevel(df$Brand, ref=as.character(Brand.expo$Group.1[which(Brand.expo$x==max(Brand.expo$x))]))
df$Region = relevel(df$Region, ref=as.character(Region.expo$Group.1[which(Region.expo$x==max(Region.expo$x))]))
df$Gas = relevel(df$Gas, ref=as.character(Gas.expo$Group.1[which(Gas.expo$x==max(Gas.expo$x))]))

# Split train / test
set.seed(564738291)
u <- runif(dim(df)[1], min = 0, max = 1)
df$train <- u < 0.7
df$test <- !(df$train)


#### Modelling
# Start with a simple model with 3 predictors.

# First simple model
m1 <- glm(ClaimNb ~ DriverAge + Region + Density, 
          data =df,
          subset = train,
          family = poisson(link = "log"),
          offset = log(Exposure))
summary(m1)

# We want to understand the potential gain of adding the other predictors to the first model. 
# For that, we add successively a factor to the equation and compute the AIC and Deviance.

glimpse(df)

# Create the grid
result_grid <- expand.grid(
  covariates = c( 0,'Power', 'CarAge', 'Brand', 'Gas'),
  AIC = NA,
  Deviance = NA)

# Iterate through the model
for(i in seq_len(nrow(result_grid))) {
  fmla <- as.formula(paste("ClaimNb ~ DriverAge + Region + Density", 
                           result_grid$covariates[i],sep = "+"))
  f <- glm(fmla,
           data = df,
           subset = train,
           family = poisson(link = "log"),
           offset = log(Exposure))
  
  #rms[v] <- RMSEP(dta$clm.count[dta$train],
  #predict(f, newdata = dta[dta$train,],
  #type = "response"))
  # print(fmla)
  result_grid$AIC[i] <- f$aic
  result_grid$Deviance[i] <- f$deviance
}
print(result_grid)

# We can represent the results on a graph.
scatter <- ggplot(result_grid, aes(x=AIC, y=Deviance)) +
  geom_point() + # Show dots
  geom_text(
    label=result_grid$covariates, 
    nudge_x = 0.25, nudge_y = 0.25, 
    check_overlap = T
  ) +
  labs(
    title = "AIC and Deviance for adding one single factor")
print(scatter)

# Adding the Brand could be considered as an asset for the model. 
# We observe that each model are performing better than the model "0", with the 3 predictors. 
# Power improves the model but less than the other variables.

### Analysis of Deviance
anova(m1, test = "Chisq")

# All the parameters are statistically significant.

### Backward stepwise regression

# Initialize a model with all predictors
backward_model <- glm(ClaimNb ~ DriverAge + Region + Density + Power + CarAge + Brand + Gas, 
                      data = df,
                      subset = train,
                      family = poisson(link = "log"),
                      offset = log(Exposure))

# Backward stepwise regression
backward_model <- step(backward_model, direction = "backward")

# As suggested by the previous graph, the variable Power does not have a significant impact on the claims frequency. 
# It has been excluded by the backward selection.
m3 <- glm(ClaimNb ~ DriverAge + Region + Density + CarAge + Brand + Gas, 
          data = df,
          subset = train,
          family = poisson(link = "log"),
          offset = log(Exposure))
summary(m3)

# Intermediary step on Region

# Another variable: Region 
with(df, table(Region, ClaimNb))

m_region <- glm(formula = ClaimNb ~ Region,
          family = poisson(link = "log"),
          data = df,
          subset = train, offset = log(Exposure))
summary(m_region)
# The "Normandie" region is not significant

# Isolate the region's name
region_name <- df %>% group_by(Region) %>% summarise(count=n())

# Run a prediction for each of the Region
# We retrieve 10 avg frequency
y=predict(m_region,newdata=
            data.frame(Region=region_name$Region,
                       Exposure=1),type="response", 
          se.fit =TRUE) # we add the CI

# Predictions and CI
pred_values <- y$fit
lower_CI <- y$fit-y$se.fit
upper_CI <- y$fit+y$se.fit

# Definition of the region for each prediction
vec_Region <-c("Centre", "Aquitaine", "Basse-Normandie", "Bretagne", "Haute-Normandie", "Ile-de-France", "Limousin", "Nord-Pas-de-Calais", "Pays-de-la-Loire", "Poitou-Charentes")

# Create the data frame
predicted_df <- data.frame(predicted_value=pred_values, Region = vec_Region, upper = upper_CI, lower = lower_CI)

#print(predicted_df)

# Create a bar plot with ggplot2
ggplot(predicted_df, aes(x = Region, y = predicted_value)) +
  geom_bar(stat = "identity",fill = "skyblue", color = "black") +
  geom_errorbar(aes(ymin = lower, ymax = upper), 
                width = 0.2, color = "red") +
  labs(title = "Claims frequency by Region", x = "Region", y = "Predicted value") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# We can attempt some grouping


df <- df %>%
  mutate(Region3 = case_when(
    Region %in% c("Basse-Normandie", "Haute-Normandie", "Bretagne", "Centre","Aquitaine","Pays-de-la-Loire", "Poitou-Charentes") ~ "Group_Ouest",
    TRUE ~ Region   # Keep other levels unchanged
  ))
df %>% group_by(Region3) %>% summarise(count=n())

# New model with new Region 
m3tris <- glm(ClaimNb ~ DriverAge + Region3 + Density + CarAge + Brand + Gas, 
              data = df,
              subset = train,
              family = poisson(link = "log"),
              offset = log(Exposure))
summary(m3tris)

# Likelhod Ratio test
lrtest(m3, m3tris)

# Chosing the new model is an improvement.


# Saving the model
saveRDS(m3tris, file = "models/GLM_clm_freq.rda")
saveRDS(m3, file = "models/GLM_clm_freq_v1.rda")


#loading the model
#model_old = readRDS("models/GLM_clm_freq.rda")

#checking whether the model has been loaded with different name
# ls()


# Prediction on the model
test <- df %>% filter(test == TRUE)

pred <- predict(m3, test, type = "response")
head(pred)


### Relativity plot
# We want to see the relativity coefficient for the variable Region.

# Aggregate exposure by category
exposure_summary <- df %>%
  group_by(Region) %>%
  summarize(Exposure = sum(Exposure), .groups = "drop")
print(exposure_summary)

# Rename the column Region as Category
exposure_summary <- exposure_summary %>% rename(Category = Region)

# Extract relativities from the model
relativities <- data.frame(
  Category = gsub("Category", "", names(coef(m3))),
  Relativity = exp(coef(m3))
)
print(relativities)

# Remove intercept and adjust category names
relativities <- relativities %>%
  filter(Category != "(Intercept)") %>% mutate(Category = gsub("Region", "", Category))

print(relativities)

#glm_model$xlevels

# Add baseline relativity (for reference)
baseline <- data.frame(Category = "Centre", Relativity = 1)
relativities <- bind_rows(baseline, relativities)
print(baseline)
print(relativities)


# Merge exposures and relativities
plot_data <- left_join(exposure_summary, relativities, by = "Category")
print(plot_data)

# Relativity graph with exposure as bars and relativity as a line
# ggplot(plot_data, aes(x = Category)) +
#   geom_bar(aes(y = Exposure), stat = "identity", fill = "skyblue", alpha = 0.7) +
#   geom_line(aes(y = Relativity * max(Exposure)), color = "blue", linewidth = 1, group = 1) +
#   geom_point(aes(y = Relativity * max(Exposure)), size = 3, color = "blue") +
#   scale_y_continuous(
#     name = "Exposure",
#     sec.axis = sec_axis(~ . / max(plot_data$Exposure), name = "Relativity")
#   ) +
#   geom_hline(yintercept = max(plot_data$Exposure), linetype = "dashed", color = "red") +
#   theme_minimal() +
#   labs(
#     title = "Exposure and Relativity Graph",
#     x = "Category",
#     caption = "Dashed line indicates baseline relativity"
#   ) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Relativity graph with relativity on primary axis and exposure on secondary axis
ggplot(plot_data, aes(x = Category)) +
  geom_line(aes(y = Relativity), color = "blue", size = 1, group = 1) +
  geom_point(aes(y = Relativity), size = 3, color = "blue") +
  geom_bar(aes(y = Exposure / max(Exposure)), stat = "identity", fill = "skyblue", alpha = 0.7) +
  scale_y_continuous(
    name = "Relativity",
    sec.axis = sec_axis(~ . * max(plot_data$Exposure), name = "Exposure")
  ) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Relativity and Exposure Graph - Region",
    x = "Category",
    caption = "Dashed line indicates baseline relativity"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


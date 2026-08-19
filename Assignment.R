library(readxl)

data.CentralPark <- read_excel(file.choose(), sheet = "CentralParkData")
data.CentralPark <- na.omit(data.CentralPark[, c("SALE_PRICE","LIVING_SQFT")])

# (a) Regression model
model.LIVING <- lm(SALE_PRICE ~ LIVING_SQFT, data = data.CentralPark)
summary(model.LIVING)

# Forecast for the 1,899 sqft home, with standard error via prediction interval
a <- 0.05
df.forecast <- data.frame("LIVING_SQFT" = 1899)

forecast0 <- predict(model.LIVING, newdata = df.forecast, interval = "prediction", level = 1 - a)
yhat      <- forecast0[, "fit"]
StErrFcst <- (forecast0[, "upr"] - forecast0[, "fit"]) / qnorm(1 - a/2)

yhat        # 586219.1
StErrFcst   # 154810.7

# (b) Zillow's final offer
fee_rate    <- 0.08
repair_cost <- 9000
c_price <- yhat * (1 - fee_rate) - repair_cost
c_price     # 530321.6

# (c) Confidence of a profit, assuming 10% appreciation
growth_rate <- 0.10
mean_W <- (1 + growth_rate) * yhat
sd_W   <- (1 + growth_rate) * StErrFcst
z <- (c_price - mean_W) / sd_W
prob_profit <- 1 - pnorm(z)
prob_profit   # 0.749


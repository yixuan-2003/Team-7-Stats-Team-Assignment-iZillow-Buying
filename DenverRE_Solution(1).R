# =============================================================================
# Class 12 Team Assignment — Question 2
# Build your own Zestimates for 626 homes
#
# Structure follows the PizzaSales solution: read the data, look at it, create
# dummy variables with ifelse(), create interaction variables by multiplying,
# fit models with lm(), compare with summary(), forecast with predict().
# =============================================================================

if (requireNamespace("readxl", quietly = TRUE)) {
  library(readxl)
  # na = c("", "NA") matters: the RM_AGE column stores the literal text "NA"
  # for homes that were never remodeled, and without this readxl reads the whole
  # column as text.
  data.Train <- as.data.frame(read_excel("DenverRE_Train.xlsx", sheet = "TrainingData",
                                         na = c("", "NA")))
  data.Test  <- as.data.frame(read_excel("DenverRE_Test.xlsx",  sheet = "TestData",
                                         na = c("", "NA")))
} else {                       # fallback if readxl is unavailable
  data.Train <- read.csv("DenverRE_Train_raw.csv", stringsAsFactors = FALSE)
  data.Test  <- read.csv("DenverRE_Test_raw.csv",  stringsAsFactors = FALSE)
}


# =============================================================================
# PART 1 — Descriptive data analysis
# =============================================================================

summary(data.Train)

# The dependent variable is strongly right-skewed, which is the reason we also
# consider a model on the logarithm of price.
hist(data.Train$SALE_PRICE, breaks = 50,
     main = "Sale Price", xlab = "$")
hist(log(data.Train$SALE_PRICE), breaks = 50,
     main = "Log of Sale Price", xlab = "log($)")

plot(data.Train$LIVING_SQFT, data.Train$SALE_PRICE,
     xlab = "Living Square Feet", ylab = "Sale Price", pch = 16, col = "grey40")

# Price per square foot is very different across neighborhoods. This is the
# reason we later add slope dummies rather than only intercept dummies.
PPSF <- data.Train$SALE_PRICE / data.Train$LIVING_SQFT
boxplot(PPSF ~ data.Train$NBHD, las = 2, cex.axis = 0.6,
        main = "Price per Living Square Foot by Neighborhood", xlab = "", ylab = "$")

table(data.Train$NBHD)
table(data.Train$PROP_CLASS)

boxplot(data.Train$SALE_PRICE ~ data.Train$PROP_CLASS,
        main = "Sale Price by Property Class", xlab = "", ylab = "$")

# RM_AGE is blank for 60.1% of the rows. This is not missing data: the data
# dictionary says a blank means the home was never remodeled. The evidence
# below confirms it — homes with a blank RM_AGE are almost new.
sum(is.na(data.Train$RM_AGE)) / nrow(data.Train)
median(data.Train$BLDG_AGE[ is.na(data.Train$RM_AGE)])
median(data.Train$BLDG_AGE[!is.na(data.Train$RM_AGE)])

# LAND_SQFT is exactly zero for every condominium, because a condominium owns
# no land. That zero is a real value, not a missing one.
table(data.Train$LAND_SQFT == 0, data.Train$PROP_CLASS)


# =============================================================================
# PART 2 — Create the new variables
#
# Everything below is done to BOTH files, in the same way, so that the
# training data and the test data have exactly the same columns.
# =============================================================================

prepare <- function(d) {
  
  # --- Make sure the numeric columns really are numeric --------------------
  # RM_AGE holds the text "NA" for homes that were never remodeled, which can
  # make the whole column read in as text. as.numeric() turns that text into a
  # genuine missing value, which is what the rest of this function expects.
  num.cols <- c("SALE_PRICE","LIVING_SQFT","FBSMT_SQFT","BSMT_AREA","LAND_SQFT",
                "GRD_AREA","BLDG_AGE","RM_AGE","BED_RMS","FULL_B","HLF_B","STORY")
  for (v in intersect(num.cols, names(d))) {
    d[[v]] <- suppressWarnings(as.numeric(d[[v]]))
  }
  stopifnot(sum(is.na(d$RM_AGE)) > 0)   # blanks must survive as real NA
  
  # --- Dummy (0/1) variable for property type ------------------------------
  d$CONDO01 <- ifelse(d$PROP_CLASS == "CONDOMINIUMS", 1, 0)
  
  # --- Remodel information -------------------------------------------------
  # A blank RM_AGE means "never remodeled", so we record that fact in a dummy
  # and build an effective age: years since the home was last new or renovated.
  d$REMOD01 <- ifelse(is.na(d$RM_AGE), 0, 1)
  d$EFF_AGE <- ifelse(is.na(d$RM_AGE), d$BLDG_AGE, d$RM_AGE)
  
  # --- Presence dummies for the structural zeros ---------------------------
  # These let the model fit a level shift for "has none at all" separately
  # from the slope on "how much".
  d$HASLAND01  <- ifelse(d$LAND_SQFT  > 0, 1, 0)
  d$HASBSMT01  <- ifelse(d$BSMT_AREA  > 0, 1, 0)
  d$HASFBSMT01 <- ifelse(d$FBSMT_SQFT > 0, 1, 0)
  
  # --- Logarithmic transformations -----------------------------------------
  # log(1 + x) rather than log(x), because many of these variables are zero
  # and log(0) is undefined.
  d$LOG_LIVING <- log(d$LIVING_SQFT)
  d$LOG_LAND   <- log(1 + d$LAND_SQFT)
  d$LOG_FBSMT  <- log(1 + d$FBSMT_SQFT)
  d$LOG_AGE    <- log(1 + d$EFF_AGE)
  
  # --- Combined room measure -----------------------------------------------
  # A half bathroom is worth roughly half a full one.
  d$TOT_BATH <- d$FULL_B + 0.5 * d$HLF_B
  
  # --- Neighborhood dummies -------------------------------------------------
  # 14 neighborhoods, so 13 dummies. Gateway / Green Valley Ranch is the
  # reference category and is deliberately left out to avoid the dummy
  # variable trap.
  d$NB_CAPITOL   <- ifelse(d$NBHD == "CAPITOL HILL",    1, 0)
  d$NB_CENTRAL   <- ifelse(d$NBHD == "CENTRAL PARK",    1, 0)
  d$NB_FIVEPTS   <- ifelse(d$NBHD == "FIVE POINTS",     1, 0)
  d$NB_HAMPDEN   <- ifelse(d$NBHD == "HAMPDEN",         1, 0)
  d$NB_HAMPDENS  <- ifelse(d$NBHD == "HAMPDEN SOUTH",   1, 0)
  d$NB_HIGHLAND  <- ifelse(d$NBHD == "HIGHLAND",        1, 0)
  d$NB_MARSTON   <- ifelse(d$NBHD == "MARSTON",         1, 0)
  d$NB_MONTBELLO <- ifelse(d$NBHD == "MONTBELLO",       1, 0)
  d$NB_MONTCLAIR <- ifelse(d$NBHD == "MONTCLAIR",       1, 0)
  d$NB_NPARKHILL <- ifelse(d$NBHD == "NORTH PARK HILL", 1, 0)
  d$NB_SLOANLAKE <- ifelse(d$NBHD == "SLOAN LAKE",      1, 0)
  d$NB_UNIONSTN  <- ifelse(d$NBHD == "UNION STATION",   1, 0)
  d$NB_WINDSOR   <- ifelse(d$NBHD == "WINDSOR",         1, 0)
  
  # --- Style dummies --------------------------------------------------------
  # The 12 style codes are grouped into 6, because four of them have fewer
  # than 15 observations and a dummy fitted on so few rows is mostly noise.
  # One-story is the reference category.
  d$ST_2STORY  <- ifelse(d$STYLE_CN %in% c("2 STORY"), 1, 0)
  d$ST_PARTIAL <- ifelse(d$STYLE_CN %in% c("1.5 STORY", "2.5 STORY",
                                           "3 STORY", "CONVERSION"), 1, 0)
  d$ST_MULTI   <- ifelse(d$STYLE_CN %in% c("BI-LEVEL", "SPLIT LEVEL",
                                           "TRI-LEVEL", "TRI-LEVEL W/B"), 1, 0)
  d$ST_ENDUNIT <- ifelse(d$STYLE_CN == "END UNIT",    1, 0)
  d$ST_MIDUNIT <- ifelse(d$STYLE_CN == "MIDDLE UNIT", 1, 0)
  
  # --- Interaction effects (slope dummies) ---------------------------------
  # Each of these is the product of a neighborhood dummy and log living area,
  # exactly like PCIxCOMP in the pizza example. They let the price PER SQUARE
  # FOOT differ across neighborhoods, not just the intercept.
  d$CAPITOLxSQFT   <- d$NB_CAPITOL   * d$LOG_LIVING
  d$CENTRALxSQFT   <- d$NB_CENTRAL   * d$LOG_LIVING
  d$FIVEPTSxSQFT   <- d$NB_FIVEPTS   * d$LOG_LIVING
  d$HAMPDENxSQFT   <- d$NB_HAMPDEN   * d$LOG_LIVING
  d$HAMPDENSxSQFT  <- d$NB_HAMPDENS  * d$LOG_LIVING
  d$HIGHLANDxSQFT  <- d$NB_HIGHLAND  * d$LOG_LIVING
  d$MARSTONxSQFT   <- d$NB_MARSTON   * d$LOG_LIVING
  d$MONTBELLOxSQFT <- d$NB_MONTBELLO * d$LOG_LIVING
  d$MONTCLAIRxSQFT <- d$NB_MONTCLAIR * d$LOG_LIVING
  d$NPARKHILLxSQFT <- d$NB_NPARKHILL * d$LOG_LIVING
  d$SLOANLAKExSQFT <- d$NB_SLOANLAKE * d$LOG_LIVING
  d$UNIONSTNxSQFT  <- d$NB_UNIONSTN  * d$LOG_LIVING
  d$WINDSORxSQFT   <- d$NB_WINDSOR   * d$LOG_LIVING
  
  # Condominiums are priced differently per square foot, and they age
  # differently, so those interactions are included as well.
  d$CONDOxSQFT <- d$CONDO01 * d$LOG_LIVING
  d$CONDOxAGE  <- d$CONDO01 * d$LOG_AGE
  
  d
}

data.Train <- prepare(data.Train)
data.Test  <- prepare(data.Test)


# =============================================================================
# PART 3 — The regression models
# =============================================================================

# The list of independent variables, written once and reused. The first group
# is the property attributes, then the neighborhood and style dummies, then
# the interaction effects.

# One variable per idea. We deliberately do NOT include both LIVING_SQFT and
# LOG_LIVING, or all three of BLDG_AGE, EFF_AGE and LOG_AGE: those are different
# ways of writing the same thing, they add nothing, and they make the individual
# coefficients impossible to interpret. HASLAND01 is also left out because it is
# exactly 1 - CONDO01.
#
#   LOG_LIVING   size of the home            LOG_AGE    how old / how long since renovation
#   LOG_FBSMT    finished basement space     REMOD01    whether it was ever renovated
#   HASBSMT01    whether there is a basement BED_RMS    bedrooms
#   LOG_LAND     size of the lot             TOT_BATH   bathrooms (half counts as 0.5)
#   CONDO01      condominium or house        STORY      how the space is arranged

vars.attributes <- "LOG_LIVING + LOG_FBSMT + HASBSMT01 + LOG_LAND +
                    LOG_AGE + REMOD01 + TOT_BATH + CONDO01"

# BED_RMS and STORY were dropped: once living area and the number of bathrooms
# are in the model, neither adds anything. Removing them slightly IMPROVES the
# cross-validated error, so keeping them would only cost degrees of freedom.

# Only the neighborhood dummies are kept here, and only for Models 1 and 2.
# The style dummies were dropped because they change cross-validated error by
# about $50, which is nothing.
vars.dummies <- "NB_CAPITOL + NB_CENTRAL + NB_FIVEPTS + NB_HAMPDEN +
                 NB_HAMPDENS + NB_HIGHLAND + NB_MARSTON + NB_MONTBELLO +
                 NB_MONTCLAIR + NB_NPARKHILL + NB_SLOANLAKE + NB_UNIONSTN +
                 NB_WINDSOR"

vars.interactions <- "CAPITOLxSQFT + CENTRALxSQFT + FIVEPTSxSQFT + HAMPDENxSQFT +
                      HAMPDENSxSQFT + HIGHLANDxSQFT + MARSTONxSQFT +
                      MONTBELLOxSQFT + MONTCLAIRxSQFT + NPARKHILLxSQFT +
                      SLOANLAKExSQFT + UNIONSTNxSQFT + WINDSORxSQFT +
                      CONDOxSQFT + CONDOxAGE"

# Model 0: living area only. This is the same specification used in Question 1,
# and it is the benchmark everything else has to beat.
model.SQFT <- lm(SALE_PRICE ~ LIVING_SQFT, data = data.Train)
summary(model.SQFT)

# Model 1: price, with all attributes and all dummies.
f1 <- as.formula(paste("SALE_PRICE ~", vars.attributes, "+", vars.dummies))
model.Level <- lm(f1, data = data.Train)
summary(model.Level)

# Model 2: the same independent variables, but with the logarithm of price as
# the dependent variable.
f2 <- as.formula(paste("log(SALE_PRICE) ~", vars.attributes, "+", vars.dummies))
model.Log <- lm(f2, data = data.Train)
summary(model.Log)

# Model 3: log of price, with the interaction effects added.
# Model 3 replaces the 13 neighborhood intercept dummies with the 15 interaction
# terms. Each neighborhood therefore gets its own price PER SQUARE FOOT rather
# than its own level. We tested keeping both; cross-validation says the separate
# intercepts add nothing once the interactions are present, so including them
# would spend 13 degrees of freedom for no gain.
f3 <- as.formula(paste("log(SALE_PRICE) ~", vars.attributes, "+", vars.interactions))
model.LogInter <- lm(f3, data = data.Train)
summary(model.LogInter)

# Are the interaction effects jointly worth including?
anova(model.Log, model.LogInter)


# =============================================================================
# PART 4 — Comparing the models
#
# The assignment grades the predictions on root mean squared error in dollars,
# so that is what we compare the models on. We cannot use the training fit for
# this, because a model with more variables always fits the training data
# better. Instead we use 10-fold cross validation: the data is split into ten
# parts, each part is predicted by a model estimated on the other nine, and
# every home is therefore predicted by a model that has never seen it.
# =============================================================================

cv.rmse <- function(form, logy, k = 10) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out = nrow(data.Train)))
  mse <- numeric(k)
  for (i in 1:k) {
    fit  <- lm(form, data = data.Train[folds != i, ])
    pred <- predict(fit, newdata = data.Train[folds == i, ])
    
    # A model fitted on log price predicts the log. Taking exp() of that gives
    # the median price, not the mean, and is biased low in dollars. Multiplying
    # by the average of exp(residual) from the training part corrects this.
    if (logy) pred <- exp(pred) * mean(exp(residuals(fit)))
    
    mse[i] <- mean((pred - data.Train$SALE_PRICE[folds == i])^2)
  }
  sqrt(mean(mse))
}

RMSE.SQFT     <- cv.rmse(SALE_PRICE ~ LIVING_SQFT, logy = FALSE)
RMSE.Level    <- cv.rmse(f1, logy = FALSE)
RMSE.Log      <- cv.rmse(f2, logy = TRUE)
RMSE.LogInter <- cv.rmse(f3, logy = TRUE)

data.frame(
  Model = c("Living area only (Question 1)",
            "Price, attributes + dummies",
            "Log price, attributes + dummies",
            "Log price, + interaction effects"),
  Variables = c(1, length(all.vars(f1)) - 1,
                length(all.vars(f2)) - 1, length(all.vars(f3)) - 1),
  CV.RMSE = round(c(RMSE.SQFT, RMSE.Level, RMSE.Log, RMSE.LogInter))
)


# =============================================================================
# PART 5 — Forecast the 626 test homes with the chosen model
# =============================================================================

model.Final <- model.LogInter

pred.log <- predict(model.Final, newdata = data.Test)

# Convert back to dollars, with the same smearing correction used above.
smear <- mean(exp(residuals(model.Final)))
PREDICTED_PRICE <- exp(pred.log) * smear

# Check the forecasts before writing them out.
length(PREDICTED_PRICE)
sum(is.na(PREDICTED_PRICE))
summary(PREDICTED_PRICE)

output <- data.frame("ID" = data.Test$ID, "PREDICTED_PRICE" = PREDICTED_PRICE)
head(output)

write.csv(output, "Predictions_Team7.csv", row.names = FALSE)

# Copy the PREDICTED_PRICE column into the yellow column of DenverRE_Test.xlsx,
# check that the IDs line up, and save the file as DenverRE_Test_Team7.xlsx.

options(width = 100)

cat("\014"); summary(model.Level)     
cat("\014"); summary(model.Log)      
cat("\014"); summary(model.LogInter) 


library(MASS)
# library(ISLR)

# fix(Boston)
data(Boston)
head(Boston)

names(Boston)

?Boston

lm.fit = lm(medv ∼ lstat)

lm.fit = lm(medv ~ lstat, data = Boston)

attach(Boston)
lm.fit = lm(medv ~ lstat)

lm.fit

summary(lm.fit)

names(lm.fit)

coef(lm.fit)

confint(lm.fit)

predict(lm.fit , data.frame (lstat = c(5, 10, 15)), interval = "confidence")

predict(lm.fit , data.frame (lstat = c(5, 10, 15)), interval = "prediction")

plot(lstat, medv)
abline(lm.fit)

plot(lstat, medv)
abline(lm.fit, lwd = 3)

plot(lstat, medv)
abline(lm.fit, lwd = 3, col = "red")

plot(lstat, medv, col ="red")

plot(lstat, medv, pch = 20)

plot(lstat, medv, pch = "+")

plot(1:20, 1:20, pch = 1:20)

par(mfrow = c(2, 2))
plot(lm.fit)

plot(predict(lm.fit), residuals(lm.fit))

plot(predict(lm.fit), rstudent(lm.fit))

plot(hatvalues(lm.fit))

which.max(hatvalues(lm.fit))

lm.fit = lm(medv ~ lstat + age, data = Boston)
summary(lm.fit)

lm.fit = lm(medv ~ ., data = Boston)
summary(lm.fit)

library(car)

vif(lm.fit)

lm.fit1 = lm(medv ~ . - age ,data = Boston)
summary(lm.fit1)

lm.fit1 = update(lm.fit, ~ . - age)

summary(lm(medv ~ lstat*age, data = Boston))

lm.fit2 = lm(medv ~ lstat + I(lstat^2))

summary(lm.fit2)

lm.fit = lm(medv ~ lstat)
anova(lm.fit, lm.fit2)

par(mfrow = c(2, 2))
plot(lm.fit2)

lm.fit5 = lm(medv ~ poly(lstat, 5))
summary(lm.fit5)

summary(lm(medv ~ log(rm), data = Boston))

# data(Carseats)
Carseats = read.csv("../../../data/Carseats.csv")

head(Carseats)

names(Carseats)

lm.fit = lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
summary(lm.fit)

attach(Carseats)
contrasts(ShelveLoc)

LoadLibraries

LoadLibraries()

LoadLibraries= function () {
  # library (ISLR )
  library (MASS )
  print (" The libraries have been loaded .")
}

LoadLibraries

LoadLibraries()


Auto = read.csv("../../../data/Auto.csv", header = T, na.strings = "?")
Auto = na.omit(Auto)
summary(Auto)

lm.fit1 = lm(mpg ~ horsepower, data = Auto)
summary(lm.fit1)

predict(lm.fit1, data.frame(horsepower=c(98)), interval="confidence")

predict(lm.fit1, data.frame(horsepower=c(98)), interval="prediction")

plot(Auto$horsepower, Auto$mpg, main = "Scatterplot of mpg vs. horsepower", xlab = "horsepower", ylab = "mpg", col = "blue")
abline(lm.fit1, col = "red")

par(mfrow = c(2, 2))
plot(lm.fit1)

pairs(Auto)

cor(subset(Auto, select = -name))

lm.fit2 = lm(mpg ~ . - name, data = Auto)
summary(lm.fit2)

par(mfrow = c(2, 2))
plot(lm.fit2)

lm.fit3 = lm(mpg ~ .*., data = Auto[, 1:8])
summary(lm.fit3)

par(mfrow = c(2, 2))
plot(log(Auto$horsepower), Auto$mpg)
plot(sqrt(Auto$horsepower), Auto$mpg)
plot((Auto$horsepower)^2, Auto$mpg)

Carseats = read.csv("../../../data/Carseats.csv", header=T, na.strings="?")

summary(Carseats)

attach(Carseats)
lm.fit4 = lm(Sales ~ Price + Urban + US)
summary(lm.fit4)

lm.fit5 = lm(Sales ~ Price + US)
summary(lm.fit5)

confint(lm.fit5)

plot(predict(lm.fit5), rstudent(lm.fit5))

par(mfrow = c(2, 2))
plot(lm.fit5)

set.seed(1)
x = rnorm(100)
y = 2 * x + rnorm(100)

lm.fit6 = lm(y ~ x + 0)
summary(lm.fit6)

lm.fit7 = lm(x ~ y + 0)
summary(lm.fit7)

n = length(x)
t = sqrt(n - 1)*(x %*% y)/sqrt(sum(x^2) * sum(y^2) - (x %*% y)^2)
as.numeric(t)

lm.fit8 = lm(y ~ x)
summary(lm.fit8)

lm.fit9 = lm(x ~ y)
summary(lm.fit9)

set.seed(1)
x = 1:100
y = 2 * x + rnorm(100, sd = 0.1)
lm.fit10 = lm(y ~ x + 0)
summary(lm.fit10)

lm.fit11 = lm(x ~ y + 0)
summary(lm.fit11)

x = 1:100
y = 100:1
lm.fit12 = lm(y ~ x + 0)
summary(lm.fit12)

lm.fit13 = lm(x ~ y + 0)
summary(lm.fit13)

set.seed(1)
x = rnorm(100)

eps = rnorm(100, sd = 0.25)

y = -1 + 0.5 * x + eps
length(y)

plot(x, y)

lm.fit14 = lm(y ~ x)
summary(lm.fit14)

plot(x, y)
abline(lm.fit14, col = "red")
abline(-1, 0.5, col = "blue")
legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1))

lm.fit15 = lm(y ~ x + I(x^2))
summary(lm.fit15)

set.seed(1)
x = rnorm(100)
eps = rnorm(100, sd = 0.0025)
y = -1 + 0.5 * x + eps
plot(x, y)

lm.fit16 = lm(y ~ x)
summary(lm.fit16)

abline(lm.fit16, col = "red")
abline(-1, 0.5, col = "blue")
legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1))

set.seed(1)
x = rnorm(100)
eps = rnorm(100, sd = 2.5)
y = -1 + 0.5 * x + eps
plot(x, y)

lm.fit17 = lm(y ~ x)
summary(lm.fit17)

abline(lm.fit17, col = "red")
abline(-1, 0.5, col = "blue")
legend("topleft", c("Least square", "Regression"), col = c("red", "blue"), lty = c(1, 1))

confint(lm.fit14)

confint(lm.fit16)

confint(lm.fit17)

set.seed(1)
x1 = runif(100)
x2 = 0.5 * x1 + rnorm(100)/10
y = 2 + 2*x1 + 0.3*x2 + rnorm(100)

cor(x1, x2)

plot(x1, x2)

lm.fit18 <- lm(y ~ x1 + x2)
summary(lm.fit18)

lm.fit19 <- lm(y ~ x1)
summary(lm.fit19)

lm.fit20 <- lm(y ~ x2)
summary(lm.fit20)

x1 = c(x1, 0.1)
x2 = c(x2, 0.8)
y = c(y, 6)

lm.fit21 = lm(y ~ x1 + x2)
summary(lm.fit21)

lm.fit22 = lm(y ~ x1)
summary(lm.fit22)

lm.fit23 = lm(y ~ x2)
summary(lm.fit23)

par(mfrow=c(2,2))
plot(lm.fit21)

par(mfrow=c(2,2))
plot(lm.fit22)

par(mfrow=c(2,2))
plot(lm.fit23)

Boston = read.csv("../../../data/Boston.csv", header=T, na.strings="?")
Boston$chas <- factor(Boston$chas, labels = c("N","Y"))
summary(Boston)

attach(Boston)
lm.zn = lm(crim~zn)
summary(lm.zn) # yes

lm.indus = lm(crim~indus)
summary(lm.indus) # yes

lm.chas = lm(crim~chas) 
summary(lm.chas) # no

lm.nox = lm(crim~nox)
summary(lm.nox) # yes

lm.rm = lm(crim~rm)
summary(lm.rm) # yes

lm.age = lm(crim~age)
summary(lm.age) # yes

lm.dis = lm(crim~dis)
summary(lm.dis) # yes

lm.rad = lm(crim~rad)
summary(lm.rad) # yes

lm.tax = lm(crim~tax)
summary(lm.tax) # yes

lm.ptratio = lm(crim~ptratio)
summary(lm.ptratio) # yes

lm.black = lm(crim~black)
summary(lm.black) # yes

lm.lstat = lm(crim~lstat)
summary(lm.lstat) # yes

lm.medv = lm(crim~medv)
summary(lm.medv) # yes

lm.all = lm(crim ~ ., data = Boston)
summary(lm.all)

# Simple regresion
x = c(coefficients(lm.zn)[2],
      coefficients(lm.indus)[2],
      coefficients(lm.chas)[2],
      coefficients(lm.nox)[2],
      coefficients(lm.rm)[2],
      coefficients(lm.age)[2],
      coefficients(lm.dis)[2],
      coefficients(lm.rad)[2],
      coefficients(lm.tax)[2],
      coefficients(lm.ptratio)[2],
      coefficients(lm.black)[2],
      coefficients(lm.lstat)[2],
      coefficients(lm.medv)[2])

# Multiple regresion
y = coefficients(lm.all)[2:14]

plot(x, y)

cor(Boston[-c(1, 4)])

lm.zn = lm(crim ~ poly(zn, 3))
summary(lm.zn) # 1, 2

lm.indus = lm(crim ~ poly(indus, 3))
summary(lm.indus) # 1, 2, 3

lm.nox = lm(crim ~ poly(nox, 3))
summary(lm.nox) # 1, 2, 3

lm.rm = lm(crim ~ poly(rm, 3))
summary(lm.rm) # 1, 2

lm.age = lm(crim ~ poly(age, 3))
summary(lm.age) # 1, 2, 3

lm.dis = lm(crim ~ poly(dis, 3))
summary(lm.dis) # 1, 2, 3

lm.rad = lm(crim ~ poly(rad, 3))
summary(lm.rad) # 1, 2

lm.tax = lm(crim ~ poly(tax, 3))
summary(lm.tax) # 1, 2

lm.ptratio = lm(crim ~ poly(ptratio, 3))
summary(lm.ptratio) # 1, 2, 3

lm.black = lm(crim ~ poly(black, 3))
summary(lm.black) # 1

lm.lstat = lm(crim ~ poly(lstat, 3))
summary(lm.lstat) # 1, 2

lm.medv = lm(crim ~ poly(medv, 3))
summary(lm.medv) # 1, 2, 3

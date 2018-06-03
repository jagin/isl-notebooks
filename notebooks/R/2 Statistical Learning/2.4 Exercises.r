
college = read.csv("../../../data/College.csv")

head(college)

rownames(college) = college[,1]
head(college[, 1:5])

college = college[,-1]
head(college[, 1:5])

summary(college)

pairs(college[,1:10])

plot(college$Private, college$Outstate,
     xlab = "Private University", ylab = "Out of State tuition in USD", main = "Outstate Tuition Plot")

Elite = rep("No", nrow(college))
Elite[college$Top10perc > 50] = "Yes"
Elite = as.factor(Elite)
college = data.frame(college, Elite)

summary(college$Elite)

plot(college$Elite, college$Outstate,
     xlab = "Elite University", ylab ="Out of State tuition in USD", main = "Outstate Tuition Plot")

par(mfrow = c(2,2))
hist(college$Books, col = 2, xlab = "Books", ylab = "Count")
hist(college$PhD, col = 3, xlab = "PhD", ylab = "Count")
hist(college$Grad.Rate, col = 4, xlab = "Grad Rate", ylab = "Count")
hist(college$perc.alumni, col = 6, xlab = "% alumni", ylab = "Count")

summary(college$PhD)

weird.phd = college[college$PhD == 103, ]
nrow(weird.phd)

rownames(weird.phd)

par(mfrow=c(1,1))
plot(college$Outstate, college$Grad.Rate)

plot(college$Accept / college$Apps, college$S.F.Ratio)

plot(college$Top10perc, college$Grad.Rate)

Auto = read.csv("../../../data/Auto.csv", header=T, na.strings="?")
Auto = na.omit(Auto)

str(Auto)

summary(Auto)

sapply(Auto[, 1:7], range)

sapply(Auto[, 1:7], mean)

sapply(Auto[, 1:7], sd)

subsetAuto = Auto[-(10:85),]
sapply(subsetAuto[, 1:7], range)

sapply(subsetAuto[, 1:7], mean)

sapply(subsetAuto[, 1:7], sd)

pairs(Auto)

plot(Auto$mpg, Auto$weight)

plot(Auto$mpg, Auto$cylinders)

plot(Auto$mpg, Auto$year)

pairs(Auto)

cor(Auto$weight, Auto$horsepower)

cor(Auto$weight, Auto$displacement)

cor(Auto$displacement, Auto$horsepower)

library(MASS)

Boston

?Boston

dim(Boston)

pairs(Boston)

par(mfrow = c(2, 2))
plot(Boston$nox, Boston$crim)
plot(Boston$rm, Boston$crim)
plot(Boston$age, Boston$crim)
plot(Boston$dis, Boston$crim)

hist(Boston$crim, breaks = 50)

pairs(Boston[Boston$crim < 20, ])

par(mfrow = c(3, 2))
plot(Boston$age, Boston$crim, main = "Older homes, more crime")
plot(Boston$dis, Boston$crim, main = "Closer to work-area, more crime")
plot(Boston$rad, Boston$crim, main = "Higher index of accessibility to radial highways, more crime")
plot(Boston$tax, Boston$crim, main = "Higher tax rate, more crime")
plot(Boston$ptratio, Boston$crim, main = "Higher pupil:teacher ratio, more crime")

par(mfrow=c(1,3))
hist(Boston$crim[Boston$crim > 1], breaks=25)
hist(Boston$tax, breaks=25)
hist(Boston$ptratio, breaks=25)

nrow(Boston[Boston$chas == 1, ])

median(Boston$ptratio)

t(subset(Boston, medv == min(Boston$medv)))

summary(Boston)

nrow(Boston[Boston$rm > 7, ])

nrow(Boston[Boston$rm > 8, ])

summary(subset(Boston, rm > 8))

summary(subset(Boston, rm > 8))

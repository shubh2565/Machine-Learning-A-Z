# Multiple Linear Regression

# Importing the dataset
dataset <- read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State <- factor(dataset$State,
                       levels = c('California', 'Florida', 'New York'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set

library(caTools)
set.seed(100)
split <- sample.split(dataset$Profit, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)


# Fitting Multiple Linear Regression to the Training set
regressor <- lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results
y_pred <- predict(regressor, newdata = test_set)


# Bacward elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor <- lm(formula = Profit ~ ., data = x)
    maxVar <- max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j <- which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x <- x[, -j]
    }
    numVars <- numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
print(backwardElimination(training_set, SL))


regressor <- lm(formula = Profit ~ R.D.Spend,
                data = training_set)

### Visualising the Training set results


library(ggplot2)
gtrain <- ggplot() +
  geom_point(aes(x = training_set$R.D.Spend, y = training_set$Profit),
             colour = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Profit vs RnD Spend (Training set)') +
  xlab('RnD Spend') +
  ylab('Profit')
print(gtrain)


### Visualising the Test set results


gtrain <- ggplot() +
  geom_point(aes(x = test_set$R.D.Spend, y = test_set$Profit),
             colour = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Profit vs RnD Spend (Training set)') +
  xlab('RnD Spend') +
  ylab('Profit')
print(gtrain)


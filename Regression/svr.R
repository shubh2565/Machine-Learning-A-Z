# SVR

# Importing the dataset
dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[2:3]


# Fitting SVR to the dataset
library(e1071)
regressor <- svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')

# Predicting a new result
y_pred <- predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results
library(ggplot2)
gtrain <- ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')
print(gtrain)


# Visualising the SVR results (for higher resolution and smoother curve)
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.1)
gtrain <- ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')
print(gtrain)
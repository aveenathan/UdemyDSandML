# Data Preprocessing

# Importing the Dataset

dataset=read.csv('Data.csv')
# dataset =dataset[ , 2:3]

# Splitting data into test and training data
set.seed(123) # sets the random number 
split =sample.split(dataset$Purchased,SplitRatio=.8)
training_set=subset(dataset, split == TRUE)
test_set=subset(dataset, split == FALSE)

#Feature Scaling , we need to exclude non numeric data and scale others
# training_set[ , 2:3] = scale(training_set[ , 2:3])
# test_set[ , 2:3] = scale(test_set[ , 2:3])

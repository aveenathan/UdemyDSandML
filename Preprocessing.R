# Data Preprocessing

# Importing the Dataset

dataset=read.csv('Data.csv')
View(dataset)

# Taking care of missing data
# if else works like IF in Google Sheets
# ave has a FUN, like a lambda function, you define it as mean(x), na.rm is used to replace by the mean
dataset$Age =ifelse(is.na(dataset$Age),ave(dataset$Age, FUN=function(x) mean(x,na.rm=TRUE)),dataset$Age)

# Encoding Categorical data

dataset$Country=factor(x=dataset$Country,levels=c('France','Spain','Germany') , labels =c(1,2,3))
dataset$Purchased=factor(x=dataset$Purchased,levels=c('No','Yes') , labels =c(0,1))
View(dataset)

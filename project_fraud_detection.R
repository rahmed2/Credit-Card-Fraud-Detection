if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
library(e1071)
library(dplyr)
library(tidyverse)
library(data.table)
library(caret)
# Data input
setwd("~/R")  # PLEASE SET THE DIRECTORY ACCORDINGLY

download.file(url="https://github.com/rahmed2/creditcardData/archive/master.zip", 
              destfile= "creditcardfraud.zip")
unzip(zipfile = "creditcardfraud.zip")

setwd("~/R/creditcardData-master") # PLEASE SET THE DIRECTORY ACCORDINGLY

unzip(zipfile = "creditcard.csv.zip")
my_data <- read.csv('creditcard.csv')
str(my_data)
setwd("..")

# Report Data
data_report<- gather(my_data, factor_key=TRUE) %>% group_by(key)%>%
  summarize(mean= mean(value), sd= sd(value), max = max(value),min = min(value))

my_data$Class <- ifelse(my_data$Class == 1, "1", "0") %>% factor(levels = c("1","0"))
# histogram of non masked data
amount_hist<-my_data %>%
  ggplot(aes(Amount)) + 
  geom_histogram(bins = 10, color = "black")
amount_time<-my_data %>%
  ggplot(aes(Time)) + 
  geom_histogram(bins = 10, color = "black")
bar_prev<-my_data %>%
  ggplot(aes(Class)) + geom_bar()
# Prevalence
K<-my_data %>% group_by(Class) %>% count() 
pr<-K$n[K$Class==1]/sum(K$n)

# Scaling non scaled data
my_data$Amount <- scale(my_data$Amount)
my_data$Time <- scale(my_data$Time)

# Creating train and test set
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(my_data$Class, times = 1, p = 0.3, list = FALSE)
train <- my_data[-test_index,]
test <- my_data[test_index,]

## ALGORITHMS

# Logistic Regression
glm_fit<- train %>% glm(Class~.,data=.,family = "binomial")
p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, "0", "1") %>% factor(levels = levels(my_data$Class))
acc1<- confusionMatrix(y_hat_logit, test$Class, positive = "1")$overall[["Accuracy"]]
F1<-F_meas(data=y_hat_logit,reference = test$Class)


x<- train[,!(colnames(my_data)=="Class")]
y<- train$Class

# LDA
fit_lda <- train(x, y, method = "lda")
fit_lda$results["Accuracy"]

y_hat_lda<-predict(fit_lda, newdata = test) %>% 
  factor(levels = levels(my_data$Class))
acc2<-confusionMatrix(y_hat_lda,test$Class)$overall[["Accuracy"]]
F2<-F_meas(data=y_hat_lda,reference = test$Class)

# QDA
fit_qda <- train(x, y, method = "qda")
fit_qda$results["Accuracy"]

y_hat_qda<-predict(fit_qda, newdata = test) %>% 
  factor(levels = levels(my_data$Class))
acc3<-confusionMatrix(y_hat_qda,test$Class)$overall[["Accuracy"]]
F3<-F_meas(data=y_hat_qda,reference = test$Class)

# Naive Bayes
fit_nb <- train(x, y, method = "naive_bayes")
fit_nb$results["Accuracy"]

y_hat_nb<-predict(fit_nb, newdata = test) %>% 
  factor(levels = levels(my_data$Class))
acc4<-confusionMatrix(y_hat_nb,test$Class)$overall[["Accuracy"]]
F4<-F_meas(data=y_hat_nb,reference = test$Class)

# PCA to try to reduce dimensions
pca<- prcomp(x)
var_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
var_explained

# ANOMALY DETECTION
# Creating train, validation and test set
a<- my_data[which(my_data$Class==0),]
b<- my_data[which(my_data$Class==1),]
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)` instead
# Train set
valid_index_ad <- createDataPartition(a$V1, times = 1, p = 0.4, list = FALSE)
train_ad <- a[-valid_index_ad,]
valid_ad <- a[valid_index_ad,]
# Valid and test set
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)` instead
test_index_ad <- createDataPartition(b$Amount, times = 1, p = 0.5, list = FALSE)
valid_ad_1 <- b[-test_index_ad,]
test_ad_1 <- b[test_index_ad,]

set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)` instead
test_ad_ad <- createDataPartition(valid_ad$Amount, times = 1, p = 0.5, list = FALSE)
valid_ad_2 <- valid_ad[-test_ad_ad,]
test_ad_2 <- valid_ad[test_ad_ad,]

valid_AD<- rbind(valid_ad_2,valid_ad_1)
test_AD<- rbind(test_ad_2,test_ad_1)

# Finding mu and sigma
train_ad$Class<- NULL
report<- gather(train_ad, factor_key=TRUE) %>% group_by(key)%>%
  summarize(mean= mean(value), sd= sd(value))
dim(report)
Prob<-NULL
for (i in 1:length(train_ad)){
  d<-dnorm(valid_AD[,i],mean = report$mean[i],sd=report$sd[i])
  Prob<- cbind(Prob,d)
}
jj<-as.vector(apply(Prob, 1, prod))

dat<-data.frame(Prob=jj,data=valid_AD$Class)
p<-dat[dat$data==1,]
acc<-0
F5<-0
for (i in 1:length(p[,1])){
  epsilon<-p$Prob[i]
  y_hat_ad<- ifelse(jj<=epsilon,"1","0") %>% factor(levels = levels(valid_AD$Class))
  acc[i]<-confusionMatrix(y_hat_ad,valid_AD$Class)$overall[["Accuracy"]]
  F5[i]<-F_meas(data=y_hat_ad,reference = valid_AD$Class)
}
  
epsilon<-p$Prob[which.max(F5)]

# on test set
Prob_new<-NULL
for (i in 1:length(train_ad)){
  d<-dnorm(test_AD[,i],mean = report$mean[i],sd=report$sd[i])
  Prob_new<- cbind(Prob_new,d)
}
j<-as.vector(apply(Prob_new, 1, prod))

y_hat<- ifelse(j<=epsilon,"1","0") %>% factor(levels = levels(test_AD$Class))
acc5<-confusionMatrix(y_hat,test_AD$Class)$overall[["Accuracy"]]
F5<-F_meas(data=y_hat,reference = test_AD$Class)

#Summary
F1_results <- data.frame(method = c("Logistic Regression","LDA","QDA","Naive Bayes", "Anomaly Detection"), 
                             F1_score = c(F1,F2,F3,F4,F5))


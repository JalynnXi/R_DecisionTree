
# Read data 
TelephoneData<-read.csv("/Users/jalynnxi/Desktop/数据挖掘/Telephone.csv", head = TRUE, fileEncoding = 'GBK')
TelephoneData

##数据结构查看与初步分析
table(TelephoneData$流失)
table(TelephoneData[,c("开通月数","流失")])
par(mfrow=c(1,2)) ##将画板变为1行2列的样式，让两张图在同一行分布
hist(TelephoneData$开通月数,main = '开通月数分布',xlab = "开通月数",ylab = "流失")
hist(TelephoneData$开通月数,main = '开通月数分布',xlab = "开通月数",ylab = "套餐类型")
 
# look at the data structure
str(TelephoneData)

# We fix the seed so that every time we run the model we do not work with different samples
set.seed(1234)  #初始化随机数发生器

# To construct any classification/regression model, we need to partition
# data into training and testing data 
# We need to randomly select instances for train and test data
# We select approximately 70% of the data for training and 30% for testing

nrow(TelephoneData)
index = sample(2, nrow(TelephoneData), replace = TRUE, prob = c(0.7,0.3))
index
TrainData = TelephoneData[index == 1, ]
TrainData
nrow(TrainData)
TestData = TelephoneData[index == 2,]
TestData
nrow(TestData)

# ******************************* DECISION TREES WITH RPART PACKAGE*******************************************************************************

# Construct a decision tree model using rpart() from "rpart" package
install.packages("rpart")
library(rpart)

#Telephone_rpart = rpart(流失~., data = TrainData,method = "class", control = rpart.control(minsplit = 10, cp = 0))
rpart.model = rpart(流失~., data = TrainData,method = "class",parms = list(split="information")) #default
rpart.model
# In "control" we can control the pruning options. 
# To learn about the settings of rpart.control, use help (write "?rpart.control" in console)

# TO plot rpart decision tree we can either use rpart.plot() function from rpart.plot package or fancyRpartPlot() function from "rattle" package.
install.packages("rpart.plot")
par(mfrow=c(1,1))
library(rpart.plot) #--- If the packages rattle is installed, it has rpart.plot in it
rpart.plot(rpart.model)
# or 
#install.packages("rattle")
#library(rattle)
#fancyRpartPlot(rpart.model)

#Predict the probablity
(predict(rpart.model))
?predict

# Look at the two-way table to check the performance of the mdoel on train data
# Check Predicted Class for TrainData
train_predict=predict(rpart.model, type = "class")  
train_predict
is.factor(train_predict)
#Check Actual Class for TrainData
TelephoneData$流失  
#or
train.predict=cbind(TrainData,train_predict)
train.predict

#Compare Predicted vs Actual：confusion matrix
(train_confusion=table(train_predict, TrainData$流失, dnn = c("Predicted", "Actual"))) 
# dnn adds the label for rows and columns
# dnn stands for dimension names

(Erorr.train=(sum(train_confusion)-sum(diag(train_confusion)))/sum(train_confusion))

# predict testing data
test_predict=predict(rpart.model,newdata = TestData,type="class")
test_predict
(test_confusion=table(test_predict, TestData$流失, dnn = c("Predicted", "Actual")))
#or
#(test_confusion=table(actural=TestData$流失,predictedclass=test_predict))

(Erorr.rpart=(sum(test_confusion)-sum(diag(test_confusion)))/sum(test_confusion))

install.packages("caret")
library(caret)
train_predict<-as.factor(train_predict)
is.factor(TrainData$流失)
TrainData$流失<-as.factor(TrainData$流失)
TestData$流失<-as.factor(TestData$流失)
sensitivity(train_predict,TrainData$流失)
specificity(train_predict,TrainData$流失)
sensitivity(test_predict,TestData$流失)
specificity(test_predict,TestData$流失)

# show CP parameters for the tree pruning
printcp(rpart.model)
plotcp(rpart.model)

# we want to prune the tree 
set.seed(1234)
rpart.prune<-prune(rpart.model,cp=0.015)
rpart.plot(rpart.prune)

# Summary of predictions on test data and training data by the pruned tree
train.predict<-predict(rpart.prune,type="class")# default data is training data
train.predict
#test.predict<-predict(rpart.prune,newdata = TestData) #default is probability
test.predict<-predict(rpart.prune,newdata = TestData,type="class")
test.predict

(test_confusion=table(actural=TestData$流失,predictedclass=test_predict))
(Erorr.rpart=(sum(test_confusion)-sum(diag(test_confusion)))/sum(test_confusion))

#train.predict=as.vector(train.predict)
sensitivity(train.predict,TrainData$流失)
specificity(train.predict,TrainData$流失)
sensitivity(test.predict,TestData$流失)
specificity(test.predict,TestData$流失)

#we can use the model to classify new data
predict(rpart.prune, TestData)

#*******************ROC***************************************
#*************************************************************
install.packages("ROCR")
library(ROCR)

#type is default means type="prob"
train.predict<-predict(rpart.prune)
test.predict<-predict(rpart.prune,newdata = TestData) 
#
train.pred = prediction(train.predict[,2],TrainData$流失) 
test.pred<-prediction(test.predict[,2],TestData$流失)

#use performance() to compute tpr and  fpr，needing return value of prediction()
train.perf<-performance(train.pred,"tpr","fpr")  
test.perf<-performance(test.pred,"tpr","fpr")  

plot(train.perf,main="ROC Curve",col = "blue", lty = 1, lwd = 3)
par(new=T)#defaulting to FALSE.如果设定为TRUE，
#If set to TRUE, the next plotting should not clean the current
plot(test.perf,main="ROC Curve",col = "red", lty = 1, lwd = 3)
#lty = 3, lwd = 3 线型，宽度
#Add straight lines to a plot (a = intercept and b = slope)
abline(a= 0, b=1)
legend("bottomright",legend=c("training","testing"),bty="n",lty=c(1,1),col=c("blue","red"))

# ***************************************** Gain Chart ****************************************** #
train.gain = performance(train.pred, "tpr", "rpp")
test.gain<-performance(test.pred,"tpr","rpp")  

plot(train.gain,main="Gain Chart",col="blue",lty = 1, lwd = 3)
par(new=T)
plot(test.gain,main="Gain chart",col = "red", lty = 1, lwd = 3)
abline(a= 0, b=1)
legend("bottomright",legend=c("training","testing"),bty="n",lty=c(1,1),col=c("blue","red"))

# **************************************** Lift Chart ******************************************** #
train.lift = performance(train.pred, "lift", "rpp")
test.lift<-performance(test.pred,"lift","rpp")  

plot(train.lift,main="Lift Curve",col="blue",lty = 1, lwd = 3)
par(new=T)
plot(test.lift,col = "red", lty = 1, lwd = 3)
legend("bottomright",legend=c("training","testing"),bty="n",lty=c(1,1),col=c("blue","red"))

# ************ DECISION TREES WITH C50 PACKAGE*************************
install.packages("C50")
install.packages("e1071")
install.packages("irr")
install.packages("lpSolve")
install.packages("vcd")
library(C50)
library(caret)
library(irr)
library(lpSolve)
library(vcd)
library(grid)

#typical decision tree with C5.0
set.seed(123)
train_sample<-sample(1000,900) # randomly select 900 samples
train<-TelephoneData[train_sample,]
test<-TelephoneData[-train_sample,]
typical.model<-C5.0(train[,-15],as.factor(train$流失)) ##训练数据框要删除分类因子向量
typical.pred<-predict(typical.model,test)
typical.confusion=table(typical.pred,test$流失,dnn = c("Predicted", "Actual"))
(Erorr.typical=(sum(typical.confusion)-sum(diag(typical.confusion)))/sum(typical.confusion))

# with boosting in C5.0()
##trials：模型的迭代次数,以10个独立决策树组合为例,winnow ：在建模之前是否对变量进行特征选择
#CF：剪枝时的置信度
boost.model<-C5.0(train[,-15],as.factor(train$流失),trials=10,control=C5.0,Control(winnow = TRUE,CF=0.25)) 
summary(boost.model)
boost.pred<-predict(boost.model,test)
boost.confusion=table(boost.pred,test$流失,dnn = c("Predicted", "Actual"))
(Erorr.typical=(sum(boost.confusion)-sum(diag(boost.confusion)))/sum(boost.confusion))


# Cross Validate and compute Kappa
set.seed(12)
folds<-createFolds(TelephoneData$流失,k=10) #根据training的laber-流失把数据集切分成10等份
cv.result<-lapply(folds,function(x){
  trainset<-TelephoneData[-x,]
  testset<-TelephoneData[x,]
  model<-C5.0(as.factor(trainset$流失)~.,data=trainset[,-15])
  pred<-predict(model,testset)
  actual<-testset$流失
  Kappa<-Kappa(table(actual,pred))
  return(Kappa)
 })
str(cv.result)
mean(unlist(cv.result))

plot(model)


# ******************************* DECISION TREES WITH PARTY PACKAGE*******************************************************************************

# construct a decision tree using ctree() from "party" package
install.packages("party")
library(party)
set.seed(123)
train_sample<-sample(1000,900) # randomly select 900 samples
train<-TelephoneData[train_sample,]
test<-TelephoneData[-train_sample,]
# Basic model
ctree.model<-ctree(流失~.,data=train)
plot(ctree.model)
plot(ctree.model,type="simple")

# Summary of predictions on test data 
ctree.predict=predict(ctree.model,newdata=test,type="response")
ctree.predict
(ctree.confusion=table(ctree.predict, test$流失, dnn = c("Predicted", "Actual")))
#or
(test_confusion=table(actural=test$流失,ctree.predict))
(Erorr.ctree=(sum(test_confusion)-sum(diag(test_confusion)))/sum(test_confusion))
#
#install.packages("gmodels")
#library(gmodels)
#ctree.confusion=CrossTable(test$流失,ctree.predict,prop.chisq=FALSE,prop.c=FALSE,prop.r=FALSE,dnn=c('actual','predicted'))


# new DATA

# Predict the "probability" (instead of classes) of prediction for test data
predict(ctree.model, newdata = newData, type = "prob")
# The predict function here returns the probablity of predictied value (With what probability or predictions are correct (it is known as confidence))


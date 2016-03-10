### Human Activy Recognition using Smartphones ###
# http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions

library(caret)
library(e1071)
library(randomForest)
library(adabag)
library(rotationForest)

# # Load in data set
feature.names <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/features.txt', header=FALSE, stringsAsFactors = FALSE) 
activity.names <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/activity_labels.txt', header=FALSE)
train.activityID <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Train/y_train.txt', header=FALSE) 
test.activityID <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Test/y_test.txt', header=FALSE)
train.subjectID <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Train/subject_id_train.txt', header=FALSE) 
test.subjectID <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Test/subject_id_test.txt', header=FALSE) 
orig.trainData <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Train/X_train.txt',header=FALSE)
orig.testData <- read.table('~/DePaul/CSC529_Advanced_Data_Mining/HAPT_Data_Set/Test/X_test.txt',header=FALSE)

# Rename column headers
colnames(train.subjectID) <- "subject"
colnames(test.subjectID) <- "subject"
#colnames(orig.trainData) <- feature.names$V1
#colnames(orig.testData) <- feature.names$V1
colnames(train.activityID) <- "activityID"
colnames(test.activityID) <- "activityID"

# Remove all features with near zero variance
#near.zero.variance <- nearZeroVar(orig.trainData, saveMetrics = T)
#reduced.trainData <- orig.trainData[, names(orig.trainData)[!near.zero.variance$nzv]]
#dim(reduced.trainData)

trainData <- cbind(train.subjectID, orig.trainData, train.activityID)
testData <- cbind(test.subjectID, orig.testData, test.activityID)
trainData$activityID <- as.factor(trainData$activityID)
testData$activityID <- as.factor(testData$activityID)

# Create data partition of traing data set
inTrain <- createDataPartition(trainData$activityID, p=0.67, list=FALSE)
myTraining <- trainData[inTrain, ]
myTesting <- trainData[-inTrain, ]
dim(myTraining); dim(myTesting)


# Find and remove highly correlated features
#correlatedPredictors <- findCorrelation(cor(myTraining[,c(-1,-563)]), cutoff = 0.8)
#reducedCorrelation <- myTraining[, -correlatedPredictors]


### RandomForest ###
preProc <- preProcess(myTraining, method = "pca", thresh = 0.85)
head(preProc$rotation)

#rfCtrl = trainControl(method = "cv", classProbs = TRUE)
rfCtrl2 =  trainControl(method = "repeatedcv", number = 10, repeats = 2, allowParallel = TRUE, classProbs=TRUE)

ptm <- proc.time()
modelRF <- train(activityID ~ V53 + V41 + V50 + V54 + V559 + V560 + 
                   V51 + V42 + V57 + V58 + V43 + V561 + V182 + 
                   V382 + V10 + V232 + V202 + V169 + V311 + V71 + 
                   V74 + V504  + V75 + V100 + V215 + V4 + V70 + V87 + V52 + 
                   V55 + V66 + V505 + V315 + V272 + V64, data = myTraining,
                 method="rf", 
                 preProcess = 'pca') 
                 #trControl = rfCtrl2)
proc.time() - ptm

modelRF.Pred <- predict(modelRF, newdata = myTesting, type="raw")
confusionMatrix(modelRF.Pred, myTesting$activityID)

modelRF.Pred.val <- predict(modelRF, newdata = testData, type="raw")
confusionMatrix(modelRF.Pred.val, testData$activityID)


varImp(modelRF, pch = 20, cex = 0.8, main =  "Variable Importance" )
imp <- varImp(modelRF)
order(imp, decreasing=TRUE)


### RandomForest on Balanced Data ###
balanced <- trainData[which(as.numeric(trainData$activityID) > 6),] #class 7-12
balanced$activityID <- as.factor(balanced$activityID)

for (i in 1:6){
  assign(paste("class",i,sep="_"), trainData[sample(which(trainData$activityID==i),100),])
}

new.balanced <- rbind(class_1, class_2, class_3, class_4, class_5, class_6, balanced)
new.balanced$activityID <- as.factor(as.integer(new.balanced$activityID))


ptm <- proc.time()
model.balanced.RF <- train(activityID ~ V53 + V41 + V50 + V54 + V559 + V560 + 
                   V51 + V42 + V57 + V58 + V43 + V561 + V182 + 
                   V382 + V10 + V232 + V202 + V169 + V311 + V71 + 
                   V74 + V504  + V75 + V100 + V215 + V4 + V70 + V87 + V52 + 
                   V55 + V66 + V505 + V315 + V272 + V64, data = new.balanced,
                 method="rf") 
#trControl = rfCtrl2)
proc.time() - ptm

modelRF.balanced.Pred <- predict(modelRF, newdata = testData, type="raw")
confusionMatrix(modelRF.balanced.Pred, testData$activityID)



### Rotation Forest ###
ptm <- proc.time()
RotForest.list <- list()
class.list <- levels(trainData$activityID)
newTrain <- trainData
newTest <- testData
#newTrain <- subset(trainData, activityID=="1"|activityID=="2"|activityID=="3"|activityID=="4"|activityID=="5"|activityID=="6")  
#newTest <- subset(testData, activityID=="1"|activityID=="2"|activityID=="3"|activityID=="4"|activityID=="5"|activityID=="6")

for (i in 1:length(class.list)){
  RotationForest.train <- newTrain
  RotationForest.test <- newTest
  
  RotationForest.train$activityID <- as.factor(ifelse(newTrain$activityID==class.list[i], class.list[i], 0))
  RotationForest.test$activityID <- as.factor(ifelse(newTest$activityID==class.list[i], class.list[i], 0))
  
  modelRotForest <- train(activityID ~ V53 + V41 + V50 + V54 + V559 + V560 + 
                            V51 + V42 + V57 + V58 + V43 + V561 + V182 + 
                            V382 + V10 + V232 + V202 + V169 + V311 + V71 + 
                            V74, data = RotationForest.train,
                          method="rotationForest")
  modelRotForestPred <- predict(modelRotForest, newdata = RotationForest.test, type="prob")
  RotForest.list[i] <- list(modelRotForestPred[,2])
}
proc.time() - ptm  # 6 classes on all data: 3518 seconds (58.5 minutes)

RotForest.table <- setNames(do.call(cbind.data.frame, RotForest.list), class.list)


RotForest.table$pred <- colnames(RotForest.table)[max.col(RotForest.table,ties.method="first")]
test.class <- as.data.frame(teData$activityID)
Rot.table <- cbind(RotForest.table, test.class)
colnames(Rot.table)[which(names(Rot.table) == "teData$activityID")] <- "class"
head(Rot.table)
table(Rot.table$pred, Rot.table$class)
confusionMatrix(Rot.table$pred, Rot.table$class)


### SVM ###
svmModel <- svm(activityID ~ V53 + V41 + V50 + V54 + V559 + V560 + 
                  V51 + V42 + V57 + V58 + V43 + V561 + V182 + 
                  V382 + V10 + V232 + V202 + V169 + V311 + V71 + 
                  V74 + V504  + V75 + V100 + V215 + V4 + V70 + V87 + V52 + 
                  V55 + V66 + V505 + V315 + V272 + V64, 
                data = myTraining, cost = 100, gamma = 1, probability=TRUE)
svmPred <- predict(svmModel, newdata = myTesting,
                   probability=TRUE)
svm.cm <- confusionMatrix(svmPred, myTesting$activityID)
svmPred.acc <- svm.cm$overall
svmPred.acc <- svmPred.acc['Accuracy'] 
svmPred.prob <- attr(svmPred, "probabilities")
svmPred.prob <- as.data.frame(unlist(svmPred.prob))


### Adabag ###
adaModel <- boosting.cv(activityID ~ V53 + V41 + V50 + V54 + V559 + V560 + 
                          V51 + V42 + V57 + V58 + V43 + V561 + V182 + 
                          V382 + V10 + V232 + V202 + V169 + V311 + V71 + 
                          V74 + V504  + V75 + V100 + V215 + V4 + V70 + V87 + V52 + 
                          V55 + V66 + V505 + V315 + V272 + V64,
                        data = myTraining, mfinal=100)
adaPred.class <- predict(adaModel, newdata = myTesting, type="class")
adaPred.pred <- predict(adaModel, newdata = myTesting, type="prob")
ada.cm <- confusionMatrix(adaPred.class, myTesting$activityID)
ada.acc <- ada.cm$overall
ada.acc <- adaPred.acc['Accuracy'] 
ada.prob <- attr(adaPred, "probabilities")
ada.prob <- as.data.frame(unlist(adaPred.prob))



### Combined Models ###
rfPred.new <- rfPred
svmPred.new <- svmPred.prob
colnames(rfPred.new) <- paste0('rf.class.', colnames(rfPred))
colnames(svmPred.new) <- paste0('svm.class.', colnames(svmPred.prob))

combined.pred <- as.data.frame(cbind(rfPred.new, svmPred.new))
combined.pred$'1' <- (combined.pred$rf.class.1 + combined.pred$svm.class.1)/2
combined.pred$'2' <- (combined.pred$rf.class.2 + combined.pred$svm.class.2)/2
combined.pred$'3' <- (combined.pred$rf.class.3 + combined.pred$svm.class.3)/2
combined.pred$'4' <- (combined.pred$rf.class.4 + combined.pred$svm.class.4)/2
combined.pred$'5' <- (combined.pred$rf.class.5)
combined.pred$'6' <- (combined.pred$rf.class.6)
combined.pred$'7' <- (combined.pred$rf.class.7 + combined.pred$svm.class.7)/2
combined.pred$'8' <- (combined.pred$rf.class.8 + combined.pred$svm.class.8)/2
combined.pred$'9' <- (combined.pred$rf.class.9 + combined.pred$svm.class.9)/2
combined.pred$'10' <- (combined.pred$rf.class.10 + combined.pred$svm.class.10)/2
combined.pred$'11' <- (combined.pred$rf.class.11 + combined.pred$svm.class.11)/2
combined.pred$'12' <- (combined.pred$rf.class.12 + combined.pred$svm.class.12)/2

combined.pred <- combined.pred[, c(25:ncol(combined.pred))]
combined.pred$pred.class <- colnames(combined.pred)[max.col(combined.pred, ties.method="first")]
combined.pred$pred.class <- as.factor(combined.pred$pred.class)
# Estimated out-of-sample-error
round((sum(myTesting$activityID != combined.pred$pred.class)) / nrow(combined.pred), 4)

combined.pred$activityID <- myTesting$activityID
combined.pred$correct <- as.numeric(combined.pred$pred.class == combined.pred$activityID)

# print accuracy
final.acc <- sum(combined.pred$correct)/nrow(combined.pred)

acc.table <- cbind(rfPred.acc, svmPred.acc, final.acc)


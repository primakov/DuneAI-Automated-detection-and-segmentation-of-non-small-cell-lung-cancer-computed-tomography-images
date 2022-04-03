#Loading libraries
library(survival)
library(survminer)
library(dynpred)
library(s2dverification)
library(fields)
library(ggplot2)

#Please specify path to the data folder
setwd("C://Users//s.primakov//Documents//My articles//lung segmentation article//2_revised_lung_article//code base//Prognostic power of segmentations//data") 
#loading data
Data_lung1 <-read.csv("Lung1_recist_volume_with_outcome.csv",header = TRUE, sep = ";", quote = "\"", dec = ",", fill = TRUE, comment.char = "")
Data_Stanford <-read.csv("Lung_stanford_recist_volume_with_outcome.csv",header = TRUE, sep = ";", quote = "\"", dec = ",", fill = TRUE, comment.char = "")

make_plots<-function(Data){
  #Converting survival time to month, limiting survival time for better visualization
  Data$surv_months <-Data$Surv_time_days/30
  for (i in 1:length(Data$surv_months)){
    if (Data$surv_months[i]>100){             
      Data$surv_months[i]<-100            
      Data$Status[i]<-0
    }
  }
  
  #Calculating median prediction for the groups
  group_vol_orig<- Data$volume_orig > median(Data$volume_orig)
  group_vol_pred<- Data$volume_predicted > median(Data$volume_predicted)
  group_recist_pred<- Data$predicted_recist > median(Data$predicted_recist)
  group_recist_orig<- Data$orig_recist > median(Data$orig_recist)
  
  #Generating the km curves
  survObjT <- Surv(Data$surv_months, Data$Status)
  
  #Generage C-index, and univariate cox HR and p-value
  res.cox_origvol <- coxph(survObjT ~ volume_orig, data = Data)
  cat('Original volume CI: ', cindex(survObjT ~ volume_orig, data = Data)$cindex, ', HR: ', summary(res.cox_origvol)$coefficients[2],', P-value: ', summary(res.cox_origvol)$coefficients[5],'\n')
  res.cox_predvol <- coxph(survObjT ~ volume_predicted, data = Data)
  cat('Predicted volume CI: ', cindex(survObjT ~ volume_predicted, data = Data)$cindex, ', HR: ', summary(res.cox_predvol)$coefficients[2],',  P-value: ', summary(res.cox_predvol)$coefficients[5],'\n')
  res.cox_origrec <- coxph(survObjT ~ orig_recist, data = Data)
  cat('Original recist CI: ', cindex(survObjT ~ orig_recist, data = Data)$cindex, ', HR: ', summary(res.cox_origrec)$coefficients[2],', P-value: ', summary(res.cox_origrec)$coefficients[5],'\n')
  res.cox_predrec <- coxph(survObjT ~ predicted_recist, data = Data)
  cat('Predicted recist CI: ', cindex(survObjT ~ predicted_recist, data = Data)$cindex, ', HR: ', summary(res.cox_predrec)$coefficients[2],', P-value: ', summary(res.cox_predrec)$coefficients[5],'\n')

  #Detailed summary of CI/HR/P-val functions
  #print(summary(res.cox_origvol))
  #print(rcorr.cens(Data$volume_orig, survObjT))
  
  km_orig_vol <- do.call(survfit,list(survObjT ~ group_vol_orig, data = Data))
  km_pred_vol <- do.call(survfit,list(survObjT ~ group_vol_pred, data = Data))
  km_orig_recist <- do.call(survfit,list(survObjT ~ group_recist_orig, data = Data))
  km_pred_recist <- do.call(survfit,list(survObjT ~ group_recist_pred, data = Data))

  #visualizing the km curves list for volume
  splots <- list()
  splots[[1]] <- ggsurvplot(km_orig_vol, risk.table = TRUE, pval = TRUE,pval.method = TRUE, conf.int = TRUE,ggtheme = theme_minimal())
  splots[[2]] <- ggsurvplot(km_pred_vol, risk.table = TRUE, pval = TRUE,pval.method = TRUE, conf.int = TRUE, ggtheme = theme_minimal())
  fig1 <- arrange_ggsurvplots(splots, print = TRUE,
                  ncol = 2, nrow = 1, risk.table.height = 0.2)
  if (FALSE) {
    # Arrange and save into pdf file
    ggsave("fig 1.pdf", fig1)
  }
  
  #Visualizing the km curves for RECIST
  splots2 <- list()
  splots2[[1]] <- ggsurvplot(km_orig_recist, risk.table = TRUE, pval = TRUE,pval.method = TRUE, conf.int = TRUE,ggtheme = theme_minimal())
  splots2[[2]] <- ggsurvplot(km_pred_recist, risk.table = TRUE, pval = TRUE,pval.method = TRUE, conf.int = TRUE, ggtheme = theme_minimal())
  fig2 <- arrange_ggsurvplots(splots2, print = TRUE,
                      ncol = 2, nrow = 1, risk.table.height = 0.2)
  
  if (FALSE) {
    # Arrange and save into pdf file
    ggsave("fig 2.pdf", fig2)
  }
  
  #Scatter plot for tumor volumes from manual and automated segmentation
  
  Data_reg_top <- subset(Data, volume_orig > 100)
  Model_top<-lm(volume_orig ~ volume_predicted, data = Data_reg_top) # linear regression model
  newx_top = seq(min(Data$volume_orig), max(Data$volume_orig),by = 0.1)
  #Confidence intervals for regression line
  conf_interval_top <- predict(Model_top, 
                               newdata=data.frame(volume_predicted=newx_top), 
                               interval="confidence", level = 0.95)
  #Regression R2 coefficient
  R2 <- format(summary(Model_top)$r.squared,digits=3)
  
  #Coefficients for regression line
  A <- summary(Model_top)$coefficients[2]
  B <- -1
  C <- summary(Model_top)$coefficients[1]
  #Estimation of the distance between every point and regression line
  x <- Data$volume_orig
  y <- Data$volume_predicted
  z <- abs(A*x + B*y + C)/sqrt(A*A + B*B)
  
  xlim=c(min(x),max(x))
  ylim=c(min(y),max(y))
  zlim=c(min(z),max(z))
  
  #Colors to interpolate Z coordinate
  nlevels = 100
  levels <- seq(zlim[1],zlim[2],length.out = nlevels)
  col <- colorRampPalette(c("blue","red"))(nlevels)  
  colz <- col[cut(z,nlevels)]
  print(levels)
  
  dev.new()
  plot(x, y,
       xlab = expression("Tumor volume from manual segmentation, mm"^"3"), 
       ylab = expression("Tumor volume from automated segmentation, mm"^"3"),
       pch = 21, 
       frame = FALSE)

  #Adding colors to scatter plot points defining distance to regression line 
  
  points(x,y,col = colz)
  image.plot( legend.only=TRUE, zlim= zlim, col = col,legend.shrink = 0.5) 
  #Adding the regression line
  abline(Model_top, col = "darkslategrey")
  #Confidence intervals for the regression line
  lines(newx_top, conf_interval_top[,2], col="darkslategrey", lty=2)
  lines(newx_top, conf_interval_top[,3], col="darkslategrey", lty=2)
  legend("topleft",c("GTV"),cex=1.0,col=c("black"),
         pt.bg=c("white"),pch=c(21))
  legend("bottomright",legend=paste("R2 = ", R2))
  legend("topright", legend=c("Regression line", "95% confidence intervals"),
         col=c("darkslategrey", "darkslategrey"), lty=1:2, cex=0.8)
  

  
return(0)
}

#Make figures for Lung1 data
lung1_plots = make_plots(Data = Data_lung1)

#Make figures for Stanford data
stanford_plots = make_plots(Data = Data_Stanford)



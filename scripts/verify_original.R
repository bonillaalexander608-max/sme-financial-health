suppressPackageStartupMessages({library(xgboost);library(recipes);library(dplyr);library(readr);library(jsonlite)})
source('R/predict.R'); b<-load_bundle()
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop('Supply the original workspace path.')
e<-new.env();load(args[1],envir=e)
# Verify the validation engine against its historical stored probabilities.
r<-e$final_fit_xgb$pre$mold$blueprint$recipe
x<-as.matrix(bake(r,new_data=e$valid_split,all_predictors()))
p<-predict(e$final_fit_xgb$fit$fit$fit,xgb.DMatrix(x,missing=0))
if(is.null(dim(p)))p<-matrix(p,ncol=3,byrow=TRUE)
error<-max(abs(p-as.matrix(e$valid_pred_xgb[c('.pred_Low','.pred_Medium','.pred_High')])))
cat('Validation probability difference',error,'\n')
stopifnot(error<1e-6)
# Pass the raw competition test through the API cleaning and actual total-data recipe.
raw<-read_delim('data/raw/Test.csv',delim=';',show_col_types=FALSE)%>%select(-ID)
labels<-character()
for(start in seq(1,nrow(raw),by=100)){
 rows<-raw[start:min(start+99,nrow(raw)),]
 payload<-fromJSON(toJSON(list(records=rows),dataframe='rows',na='null',auto_unbox=TRUE),simplifyVector=FALSE)
 out<-predict_records(payload,b)
 labels<-c(labels,vapply(out$predictions,function(x)x$class,character(1)))
}
match_rate<-mean(labels==as.character(e$y$Target))
cat('Total-data submission label agreement',match_rate,'\n');stopifnot(match_rate==1)
result<-list(validation_probability_max_absolute_difference=error,competition_test_rows=nrow(raw),submission_label_agreement=match_rate,smoke_tests='see tests/smoke.R and tests/http_smoke.py')
write_json(result,'reports/tables/api_verification.json',pretty=TRUE,auto_unbox=TRUE)

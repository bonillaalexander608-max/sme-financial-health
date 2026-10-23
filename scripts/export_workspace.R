# Explicit export only: never source the experimental tuning scripts.
# Rscript scripts/export_workspace.R /path/to/supplied.RData
suppressPackageStartupMessages({library(xgboost);library(recipes);library(dplyr)})
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop('Supply the path to your original R workspace.')
e <- new.env();load(args[1],envir=e)
wf <- e$final_fit_total
recipe <- wf$pre$mold$blueprint$recipe
booster <- wf$fit$fit$fit
classes <- wf$fit$fit$lvl
if(!inherits(recipe,'recipe') || !inherits(booster,'xgb.Booster') || !identical(classes,c('Low','Medium','High'))) stop('Expected no-ID fitted workflow is absent or incompatible.')
probe <- recipes::bake(recipe,new_data=e$submision,recipes::all_predictors())
xgboost::xgb.save(booster,'models/booster.ubj')
saveRDS(list(model_id='xgb-no-id-989',recipe=recipe,features=names(probe),classes=classes,missing_value=0),'models/preprocessing.rds')
writeLines(capture.output(sessionInfo()),'models/export_session.txt')

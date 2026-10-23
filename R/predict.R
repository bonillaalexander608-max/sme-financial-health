suppressPackageStartupMessages({library(recipes);library(xgboost);library(jsonlite)})
project_root <- Sys.getenv('FHI_PROJECT_ROOT','.')
source(file.path(project_root,'R/cleaning.R'))
load_bundle <- function(path = file.path(project_root,'models')) {
  info <- readRDS(file.path(path,'preprocessing.rds'))
  info$booster <- xgboost::xgb.load(file.path(path,'booster.ubj'))
  info$schema <- jsonlite::fromJSON(file.path(project_root,'api/schema.json'),simplifyVector=FALSE)
  info
}
input_error <- function(message) stop(structure(list(message=message,call=NULL),class=c('input_error','error','condition')))
validate_records <- function(payload,bundle) {
  if (!is.list(payload) || !identical(names(payload),'records')) input_error('Body must contain only records: an array of objects.')
  records <- payload$records
  if (!is.list(records) || !is.null(names(records)) || length(records)<1L || length(records)>100L) input_error('Supply between 1 and 100 records.')
  fields <- bundle$schema$fields
  rows <- lapply(seq_along(records), function(i) {
    row <- records[[i]]
    if (!is.list(row) || is.null(names(row)) || anyDuplicated(names(row))) input_error(sprintf('Record %d must be an object with unique field names.',i))
    if (!setequal(names(row),names(fields))) input_error(sprintf('Record %d fields do not match GET /schema.',i))
    for (key in names(fields)) {
      rule <- fields[[key]]; value <- row[[key]]
      if (is.null(value)) {
        if (!isTRUE(rule$nullable)) input_error(paste(key,'cannot be null.'))
        row[key] <- list(if (rule$type=='number') NA_real_ else NA_character_)
      } else if (rule$type=='number') {
        if (!is.numeric(value) || length(value)!=1L || !is.finite(value) || value<0) input_error(paste(key,'must be a finite nonnegative number or null.'))
      } else {
        if (!is.character(value) || length(value)!=1L || !(value %in% unlist(rule$enum))) input_error(paste('Unsupported category for',key))
      }
    }
    as.data.frame(row[names(fields)],stringsAsFactors=FALSE)
  })
  dplyr::bind_rows(rows)
}
predict_records <- function(payload,bundle) {
  raw <- validate_records(payload,bundle)
  cleaned <- limpiar_datos_empresa(raw)
  baked <- recipes::bake(bundle$recipe,new_data=cleaned,recipes::all_predictors())
  if (!identical(names(baked),bundle$features)) stop('Encoded feature order differs from the saved booster contract.')
  mat <- as.matrix(baked)
  if (!is.numeric(mat) || any(!is.finite(mat))) input_error('Record could not be preprocessed into finite numeric features.')
  dm <- xgboost::xgb.DMatrix(mat,missing=bundle$missing_value)
  probabilities <- predict(bundle$booster,dm)
  if (is.null(dim(probabilities))) probabilities <- matrix(probabilities,ncol=length(bundle$classes),byrow=TRUE)
  if (!identical(dim(probabilities),c(nrow(raw),length(bundle$classes))) || any(!is.finite(probabilities)) || any(abs(rowSums(probabilities)-1)>1e-5)) stop('Invalid model probability output.')
  colnames(probabilities)<-bundle$classes
  labels <- bundle$classes[max.col(probabilities,ties.method='first')]
  list(model_id=bundle$model_id,predictions=lapply(seq_len(nrow(raw)),function(i) list(row=i,class=labels[i],probabilities=as.list(probabilities[i,]))))
}

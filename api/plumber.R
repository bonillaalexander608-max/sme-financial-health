source(file.path(Sys.getenv('FHI_PROJECT_ROOT','..'),'R/predict.R'))
bundle <- load_bundle()
#* Service readiness
#* @get /health
#* @serializer unboxedJSON
function() list(status='ok',model_id=bundle$model_id,classes=bundle$classes)
#* Input fields and allowed categories
#* @get /schema
#* @serializer unboxedJSON
function() bundle$schema
#* Predict financial health for 1–100 business records
#* @post /predict
#* @parser json_raw
#* @serializer unboxedJSON
function(req,res) {
  tryCatch({
    if (nchar(req$postBody,type='bytes')>1000000) {res$status<-413;return(list(error='Request exceeds 1 MB.'))}
    body <- tryCatch(jsonlite::fromJSON(req$postBody,simplifyVector=FALSE),error=function(e) input_error('Malformed JSON body.'))
    predict_records(body,bundle)
  },input_error=function(e) {res$status<-400;list(error=conditionMessage(e))},error=function(e) {res$status<-500;list(error='Prediction failed; check the model artifact and server diagnostics.')})
}

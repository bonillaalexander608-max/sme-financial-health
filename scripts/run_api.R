# Run from repository root: Rscript scripts/run_api.R
port <- as.integer(Sys.getenv('PORT','8000'))
host <- Sys.getenv('HOST','127.0.0.1')
Sys.setenv(FHI_PROJECT_ROOT=normalizePath('.'))
plumber::register_parser('json_raw',function() function(value,...) list(),fixed='application/json')
plumber::plumb('api/plumber.R')$run(host=host,port=port)

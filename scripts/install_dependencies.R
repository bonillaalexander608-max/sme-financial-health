packages <- c('xgboost','recipes','dplyr','forcats','jsonlite','plumber')
missing <- setdiff(packages,rownames(installed.packages()))
if(length(missing)) install.packages(missing,repos='https://cloud.r-project.org')

if(packageVersion('xgboost') < '3.2.1.1') install.packages('xgboost',repos='https://cloud.r-project.org')

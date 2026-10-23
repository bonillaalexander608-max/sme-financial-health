library(readxl)
library(readxl)
library(tidymodels)
library(tidyverse)
Train <- read_csv("Train.csv")
submision <- read_delim("Test.csv", delim = ";", 
                        escape_double = FALSE, trim_ws = TRUE)
limpiar_datos_empresa <- function(df) {
  
  df_limpio <- df %>%
    
    # 1. Actitudes: Yes vs No
    mutate(across(
      starts_with("attitude_"),
      ~ ifelse(.x == "Yes", "Yes", "No")
    )) %>%
    
    # 2. Percepción y cumplimiento: Yes / No / Unknown
    mutate(across(
      c(starts_with("perception_"),
        compliance_income_tax,
        covid_essential_service),
      ~ case_when(
        .x == "Yes" ~ "Yes",
        .x == "No"  ~ "No",
        TRUE        ~ "Unknown"
      )
    )) %>%
    
    # 3. Posesión y uso: Yes / No / Used_to / Unknown
    mutate(across(
      c(starts_with("has_"),
        starts_with("uses_"),
        motor_vehicle_insurance,
        medical_insurance,
        funeral_insurance),
      ~ case_when(
        .x %in% c("Have now", "Yes") ~ "Yes",
        .x %in% c("Used to have but don't have now",
                  "Used to have but don’t have now") ~ "Used_to",
        .x %in% c("Never had", "No") ~ "No",
        TRUE ~ "Unknown"
      )
    )) %>%
    
    # 4. Frecuencia: Always / Sometimes / No / Unknown
    mutate(across(
      c(offers_credit_to_customers,
        keeps_financial_records),
      ~ case_when(
        .x == "Yes, always"    ~ "Always",
        .x == "Yes, sometimes" ~ "Sometimes",
        .x == "No"             ~ "No",
        TRUE                   ~ "Unknown"
      )
    )) %>%
    
    # 5. Sexo del dueño
    mutate(owner_sex = case_when(
      owner_sex == "Male"   ~ "Male",
      owner_sex == "Female" ~ "Female",
      TRUE                  ~ "Unknown"
    )) %>%
    
    # 6. Convertir strings a factor
    mutate(across(where(is.character), as.factor)) %>%
    
    # 7. NA explícitos en categóricas
    mutate(across(
      where(is.factor),
      ~ forcats::fct_explicit_na(.x, na_level = "Unknown")
    ))
  
  return(df_limpio)
}
Train<-Train%>%select(-ID)
set.seed(2026)

split <- initial_split(
  Train,
  prop = 0.8,
  strata = Target
)


train_split <- training(split)
valid_split <- testing(split)
train_split<-limpiar_datos_empresa(train_split)
valid_split<-limpiar_datos_empresa(valid_split)
submision<-limpiar_datos_empresa(submision)
total<-rbind(train_split,valid_split)
#pasando  a facror la variable objetivo 
train_split$Target <- factor(
  train_split$Target,
  levels = c("Low", "Medium", "High")
)
valid_split$Target <- factor(
  valid_split$Target,
  levels = c("Low", "Medium", "High")
)
total$Target <- factor(
  total$Target,
  levels = c("Low", "Medium", "High")
)


clasificacion_receta <- recipe(Target ~ ., data = train_split) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_mutate(
    informal_finance = case_when(
      uses_informal_lender == "Yes" ~ "Severe",
      uses_friends_family_savings == "Yes" ~ "Mild",
      TRUE ~ "None"
    ),owner_age_group = case_when(
      owner_age < 30  ~ "Young",
      owner_age <= 50 ~ "Prime",
      owner_age > 50  ~ "Senior"
    ),
    expense_ratio =
      business_expenses / (business_turnover+1 ),
    
    income_pressure =
      personal_income / (business_expenses+1),
    
    business_scale =
      business_turnover / pmax(business_age_years, 0.25),
    
    formal_finance =
      if_else(
        has_internet_banking == "Yes" |
          has_credit_card == "Yes" |
          has_loan_account == "Yes"|
          has_insurance=="Yes"|
          has_debit_card=="Yes",
        "Yes",
        "No"
      ),
    riesgo_cliente = if_else(
      (offers_credit_to_customers %in% c("Always", "Sometimes")) & 
        current_problem_cash_flow == "Yes", 
      "Yes", "No"
    ),
    
    record_discipline = case_when(
      keeps_financial_records == "Always"    ~ "High",
      keeps_financial_records == "Sometimes" ~ "Medium",
      TRUE                                   ~ "Low"
    ))%>%step_log(
    personal_income,
    business_turnover,
    business_expenses,
    offset = 1
  ) %>%step_rm(uses_informal_lender,uses_friends_family_savings)%>%
  step_string2factor(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.025)%>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())%>%
  step_zv(all_predictors())%>%step_corr(all_numeric_predictors(), threshold = 0.8)


library(xgboost)
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune(),
  stop_iter = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(clasificacion_receta) %>%   
  add_model(xgb_spec)

xgb_grid <- grid_space_filling(
  trees(range = c(600, 1200)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),   
  loss_reduction(),
  min_n(),
  sample_size = sample_prop(),
  stop_iter(),
  size = 75
)

set.seed(2026)
folds<-vfold_cv(train_split,v=5,strata = Target)
metricas <- metric_set(
  roc_auc,
  pr_auc,
  mn_log_loss,
  accuracy)

set.seed(2026)
xgb_tune <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metricas,
  control = control_grid(save_pred = TRUE,verbose = TRUE  ))

show_best(xgb_tune, metric = "roc_auc", n = 5)
show_best(xgb_tune, metric = "mn_log_loss", n = 5)
best_params_xgb <- tibble(
  trees = 989,
  min_n=2, tree_depth=6, learn_rate=0.0106,loss_reduction=0.000235,sample_size=0.976,stop_iter=5)


best_params_xgb
final_wf_xgb<- finalize_workflow(
  xgb_wf,
  best_params_xgb
)
final_fit_xgb<- fit(final_wf_xgb, data = train_split)

train_pred_xgb <- predict(
  final_fit_xgb,
  new_data = train_split,
  type = "prob")%>%bind_cols(train_split %>% select(Target))

valid_pred_xgb<-predict(
  final_fit_xgb,
  new_data = valid_split,
  type = "prob")%>%bind_cols(valid_split %>% select(Target))
levels_order <- c("Low", "Medium", "High")

train_pred_xgb<- train_pred_xgb%>%
  mutate(Target = factor(Target, levels = levels_order))

valid_pred_xgb<- valid_pred_xgb%>%
  mutate(Target = factor(Target, levels = levels_order))
######metricas
roc_auc(
  train_pred_xgb,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)

mn_log_loss(
  train_pred_xgb,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)



roc_auc(
  valid_pred_xgb,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)
mn_log_loss(
  valid_pred_xgb,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)
pred_cols<-c('.pred_Low','.pred_Medium','.pred_High')
valid_pred_xgb <- valid_pred_xgb %>%
  mutate(
    .pred_class = factor(
      pred_cols[max.col(select(., all_of(pred_cols)))],
      levels = pred_cols,
      labels = levels(Target)
    )
  )
conf_mat(
  valid_pred_xgb,
  truth = Target,
  estimate = .pred_class
) %>%autoplot(type = "heatmap")

#reentranando con el total 
final_wf_xgb<- finalize_workflow(
  xgb_wf,
  best_params_xgb
)
final_fit_total<- fit(final_wf_xgb, data = total)
ID<-submision$ID
submision<-submision%>%select(-ID)
sub_xgb_1<-predict(final_fit_xgb,new_data = submision)
sub_xgb_1<-sub_xgb_1%>%transmute(ID=ID,Target=.pred_class)
y<-predict(final_fit_total,new_data = submision)
y<-y%>%transmute(ID=ID,Target=y$.pred_class)
write.csv(y,"xgb_7_folds_total.csv",row.names = F)
########ensamblando modelo 
install.packages("fastshap")
install.packages("shapviz")
library(fastshap)
library(shapviz)

model <- extract_fit_engine(final_fit_total)

# preparar matriz de predictores
X <- bake(
  prep(clasificacion_receta),
  new_data = train_split
) %>%
  select(-Target) %>%
  as.data.frame()

# función de predicción (probabilidad de clase "High")
pred_wrapper <- function(object, newdata) {
  
  preds <- predict(object, as.matrix(newdata))
  
  # convertir a matriz (3 clases)
  preds <- matrix(preds, ncol = 3, byrow = TRUE)
  
  # devolver probabilidad de la clase High
  preds[,3]
}

set.seed(123)

shap_values <- fastshap::explain(
  object = model,
  X = X,
  pred_wrapper = pred_wrapper,
  nsim = 10
)

# visualización
sv <- shapviz(shap_values, X = X)

plot(sv)

library(xgboost)

shap_values <- predict(
  model,
  as.matrix(X),
  predcontrib = TRUE
)
dim(shap_values)
shap_array<- shap_values[,3,]
shap_array <- shap_array[, -ncol(shap_array)]
library(shapviz)
colnames(shap_array) <- colnames(X)
sv <- shapviz(
  shap_array,
  X = X
)
sv_importance(sv, kind = "beeswarm", max_display = 15)


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
      business_expenses / pmax(business_turnover,1 ),
    
    income_pressure =
      personal_income / pmax(business_expenses,1),
    
    business_scale =
      business_turnover / pmax(business_age_years,1),
    
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
    
    record_discipline = case_when(
      keeps_financial_records == "Always"    ~ "High",
      keeps_financial_records == "Sometimes" ~ "Medium",
      TRUE                                   ~ "Low"
    ))%>%step_rm(uses_informal_lender,uses_friends_family_savings, owner_age)%>%
  step_string2factor(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.02)%>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%step_zv(all_predictors())%>%
  step_normalize(all_numeric_predictors())

library(parsnip)

mlp_spec <- mlp(
  hidden_units = tune(),   # tamaño de la capa oculta
  penalty = tune(),        # L2 regularization
  epochs = tune()          # número de epochs
) %>%
  set_mode("classification") %>%
  set_engine("nnet", trace = FALSE)

wf_nn <- workflow() %>%
  add_recipe(clasificacion_receta) %>%
  add_model(mlp_spec)

set.seed(2026)

folds <- vfold_cv(
  train_split,
  v = 5,
  strata = Target
)

library(dials)

param_grid <- grid_regular(
  hidden_units(range = c(8, 32)),
  penalty(range = c(-5, -1)),   # 1e-5 a 1e-1
  epochs(range = c(50, 150)),
  levels = 4
)

library(yardstick)

metricas <- metric_set(
  accuracy,
  mn_log_loss,
  kap
)

set.seed(2026)

tuned_nn <- tune_grid(
  wf_nn,
  resamples = folds,
  grid = param_grid,
  metrics = metricas,
  control = control_grid(save_pred = TRUE)
)

best_nn <- select_best(
  tuned_nn,
  metric = "mn_log_loss"
)
best_nn
final_nn <- finalize_workflow(
  wf_nn,
  best_nn
)

fit_nn <- fit(final_nn, data = train_split)

pred_valid <- predict(fit_nn, valid_split, type = "prob") %>%
  bind_cols(
    predict(fit_nn, valid_split),
    valid_split %>% select(Target)
  )

metricas(pred_valid, truth = Target, estimate = .pred_class)

mn_log_loss(pred_valid, truth = Target,
            .pred_Low, .pred_Medium, .pred_High)
conf_mat(
  pred_valid,
  truth = Target,
  estimate = .pred_class
)
#reenetrenado
final_nn <- finalize_workflow(
  wf_nn,
  best_nn
)

fit_final_nn <- fit(
  final_nn,
  data = total
)
ID_sub<-submision$ID
submision<-submision%>%select(-ID)
pred_sub_1<-predict(fit_final_nn,new_data = submision)
pred_sub_1<-pred_sub_1%>%transmute(ID=ID_sub,Target=pred_sub$.pred_class)
write.csv(pred_sub_1,"sub_NN.csv",row.names = F)
#creando fronteras

oof_preds <- tuned_nn %>%
  collect_predictions() %>%
  filter(.config == best_nn$.config)
oof_preds <- oof_preds %>%
  mutate(
    score =
      0 * .pred_Low +
      1 * .pred_Medium +
      2 * .pred_High
  )
t1 <- quantile(oof_preds$score[oof_preds$Target == "Low"], 0.90)
t2 <- quantile(oof_preds$score[oof_preds$Target == "Medium"], 0.90)

pred_sub_prob <- predict(
  fit_final_nn,
  submision,
  type = "prob"
)
pred_sub <- pred_sub_prob %>%
  mutate(
    score =
      0 * .pred_Low +
      1 * .pred_Medium +
      2 * .pred_High
  )
pred_sub <- pred_sub %>%
  mutate(
    Target = case_when(
      score <= t1 ~ "Low",
      score <= t2 ~ "Medium",
      TRUE        ~ "High"
    )
  )
submission_df <- tibble(
  ID = ID_sub,
  Target = pred_sub$Target
)
write.csv(submission_df,'sub_nn_fronteras_oof.csv',row.names = F)
table(pred_sub$Target)
table(pred_sub_1)

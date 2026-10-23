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

library(tidymodels)
library(tidyverse)

set.seed(2026)

split <- initial_split(
  Train,
  prop = 0.8,
  strata = Target
)

submision <- read_delim("Test.csv", delim = ";", 
                        escape_double = FALSE, trim_ws = TRUE)

train_split <- training(split)
valid_split <- testing(split)
train_split<-limpiar_datos_empresa(train_split)
valid_split<-limpiar_datos_empresa(valid_split)
submision<-limpiar_datos_empresa(submision)
total<-rbind(train_split,valid_split)

library(tidymodels)
library(tidyverse)

clasificacion_receta <- recipe(Target ~ ., data = train_split) %>%
  update_role(ID, new_role = "id") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_log(
    personal_income,
    business_turnover,
    business_expenses,
    offset = 1
  ) %>%
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
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%step_zv(all_predictors())%>%
  step_normalize(all_numeric_predictors())



receta_prep <- prep(
  clasificacion_receta,
  training = train_split,
  verbose = TRUE
)

juice(receta_prep)
glimpse(juice(receta_prep))



logit_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


logit_wf <- workflow() %>%
  add_recipe(clasificacion_receta) %>%
  add_model(logit_spec)

set.seed(2026)

folds <- vfold_cv(
  train_split,
  v = 5,
  strata = Target
)
metricas <- metric_set(
  roc_auc,
  pr_auc,
  mn_log_loss,
  accuracy
)
grid_logit <- grid_regular(
  penalty(range = c(-4, 0)),   # 1e-4 a 1
  mixture(range = c(0, 1)),
  levels = c(20, 5)
)
set.seed(2026)

tune_logit <- tune_grid(
  logit_wf,
  resamples = folds,
  grid = grid_logit,
  metrics = metricas,
  control = control_grid(
    save_pred = TRUE,
    verbose = TRUE
  )
)
show_best(tune_logit, metric = "roc_auc", n = 15)
autoplot(tune_logit)
show_best(tune_logit, metric = "mn_log_loss", n = 5)


best_params <- tibble(
  penalty = 0.00113,
  mixture = 0.75
)


best_params
final_wf <- finalize_workflow(
  logit_wf,
  best_params
)
final_fit <- fit(final_wf, data = train_split)

train_pred <- predict(
  final_fit,
  new_data = train_split,
  type = "prob"
)%>%bind_cols(train_split %>% select(Target))

valid_pred<-predict(
  final_fit,
  new_data = valid_split,
  type = "prob")%>%bind_cols(valid_split %>% select(Target))
levels_order <- c("Low", "Medium", "High")
train_pred <- train_pred %>%
  mutate(Target = factor(Target, levels = levels_order))

valid_pred <- valid_pred %>%
  mutate(Target = factor(Target, levels = levels_order))

#####################
roc_auc(
  train_pred,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)

mn_log_loss(
  train_pred,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)

accuracy(
  train_pred,
  truth = Target,
  estimate = .pred_class
)

roc_auc(
  valid_pred,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)
mn_log_loss(
  valid_pred,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)


accuracy(
  valid_pred,
  truth = Target,
  estimate = .pred_class
)
pred_cols <- c(".pred_Low", ".pred_Medium", ".pred_High")

train_pred <- train_pred %>%
  mutate(
    .pred_class = factor(
      pred_cols[max.col(select(., all_of(pred_cols)))],
      levels = pred_cols,
      labels = levels(Target)
    )
  )

valid_pred <- valid_pred %>%
  mutate(
    .pred_class = factor(
      pred_cols[max.col(select(., all_of(pred_cols)))],
      levels = pred_cols,
      labels = levels(Target)
    )
  )

conf_mat(
  valid_pred,
  truth = Target,
  estimate = .pred_class
) %>%
  summary()

################################matriz de confusion
conf_mat(
  train_pred,
  truth = Target,
  estimate = .pred_class
) %>%
  summary()
conf_mat(
  valid_pred,
  truth = Target,
  estimate = .pred_class
) %>%
  summary()
conf_mat(
  valid_pred,
  truth = Target,
  estimate = .pred_class
) %>%
  autoplot(type = "heatmap")

###################################

train_split_1 <- train_split %>%
  mutate(ID = as.character(ID))

valid_split_1<- valid_split %>%
  mutate(ID = as.character(ID))

total <- bind_rows(train_split_1, valid_split_1)
clasificacion_receta_final <- recipe(Target ~ ., data = total) %>%
  update_role(ID, new_role = "id") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_log(
    personal_income,
    business_turnover,
    business_expenses,
    offset = 1
  ) %>%
  step_mutate(
    informal_finance = case_when(
      uses_informal_lender == "Yes" ~ "Severe",
      uses_friends_family_savings == "Yes" ~ "Mild",
      TRUE ~ "None"
    ),
    owner_age_group = case_when(
      owner_age < 30  ~ "Young",
      owner_age <= 50 ~ "Prime",
      owner_age > 50  ~ "Senior"
    ),
    expense_ratio =
      business_expenses / pmax(business_turnover, 1),
    income_pressure =
      personal_income / pmax(business_expenses, 1),
    business_scale =
      business_turnover / pmax(business_age_years, 1),
    formal_finance =
      if_else(
        has_internet_banking == "Yes" |
          has_credit_card == "Yes" |
          has_loan_account == "Yes" |
          has_insurance == "Yes" |
          has_debit_card == "Yes",
        "Yes", "No"
      ),
    record_discipline = case_when(
      keeps_financial_records == "Always"    ~ "High",
      keeps_financial_records == "Sometimes" ~ "Medium",
      TRUE                                   ~ "Low"
    )
  ) %>%
  step_rm(uses_informal_lender, uses_friends_family_savings, owner_age) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

final_model <- finalize_model(
  logit_spec,
  best_params
)

final_wf_full <- workflow() %>%
  add_recipe(clasificacion_receta_final) %>%
  add_model(final_model)   # el modelo ya finalizado

final_wf_full <- workflow() %>%
  add_recipe(clasificacion_receta_final) %>%
  add_model(final_model)

final_fit_full <- fit(final_wf_full, data = total)

sub_pred <- predict(
  final_fit_full,
  new_data = submision,
  type = "prob"
) %>%
  bind_cols(submision %>% select(ID)) %>%
  mutate(
    Target = colnames(select(., starts_with(".pred_")))[
      max.col(select(., starts_with(".pred_")))
    ]
  ) %>%select(ID, Target)

sub_pred<-sub_pred%>%mutate(Target=case_when(Target==".pred_High"~"High",
                                             Target==".pred_Low"~"Low",
                                             TRUE~"Medium"))
table(sub_pred$Target)
################
y<-predict(final_fit,new_data = submision,type="prob")%>%
  bind_cols(submision %>% select(ID)) %>%
  mutate(
    Target = colnames(select(., starts_with(".pred_")))[
      max.col(select(., starts_with(".pred_")))
    ]
  ) %>%select(ID, Target)
y<-y%>%mutate(Target=case_when(Target==".pred_High"~"High",
                                             Target==".pred_Low"~"Low",
                                             TRUE~"Medium"))
df<-predict(final_fit_full,new_data = submision,type = "prob")
predict_with_thresholds <- function(df,
                                    th_high = 0.55,
                                    th_low  = 0.55) {
  df %>%
    mutate(
      Target_pred = case_when(
        .pred_High >= th_high ~ "High",
        .pred_Low  >= th_low  ~ "Low",
        TRUE                  ~ "Medium"
      )
    )
}
df<-predict_with_thresholds(df)
table(df$Target_pred)
df<-df%>%transmute(ID=submision$ID,Target=Target_pred)



write.csv(df,"Submision_R_multimodal_3.csv",row.names = F)
train_split<-bake(receta_prep,new_data = train_split)
valid_split<-bake(receta_prep,new_data = valid_split)
submision<-bake(receta_prep,new_data = submision)
write.csv(train_split,"train_split.csv",row.names = F)
write.csv(valid_split,"valid_split.csv",row.names = F)
write.csv(total,"total.csv",row.names = F)
write.csv(submision,"submision.csv",row.names = F)

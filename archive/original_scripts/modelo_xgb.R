
clasificacion_receta <- recipe(Target ~ ., data = train_split) %>%
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
  step_other(all_nominal_predictors(), threshold = 0.02)%>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%step_zv(all_predictors())%>%
  step_normalize(all_numeric_predictors())

library(xgboost)
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
xgb_wf <- workflow() %>%
  add_recipe(clasificacion_receta) %>%   
  add_model(xgb_spec)

xgb_grid <- grid_space_filling(
  trees(range = c(300, 800)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),   # log10
  loss_reduction(),
  min_n(),
  sample_size = sample_prop(),
  size = 30
)

set.seed(2026)
xgb_tune <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metricas,
  control = control_grid(save_pred = TRUE)
)

show_best(xgb_tune, metric = "roc_auc", n = 5)
show_best(xgb_tune, metric = "mn_log_loss", n = 5)
best_params_xgb <- tibble(
  trees = 748,
  min_n=4, tree_depth=7, learn_rate=0.00924,loss_reduction=0.00000000961,sample_size=0.410
)


best_params_xgb
final_wf_xgb<- finalize_workflow(
  xgb_wf,
  best_params_xgb
)
final_fit_xgb<- fit(final_wf_xgb, data = train_split)

train_pred_xgb <- predict(
  final_fit_xgb,
  new_data = train_split,
  type = "prob"
)%>%bind_cols(train_split %>% select(Target))

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
sub_xgb<-predict(final_fit_xgb,new_data = submision)
sub_xgb<-sub_xgb%>%transmute(ID=submision$ID,Target=.pred_class)
########ensamblando modelo 
cols <- c(".pred_Low", ".pred_Medium", ".pred_High")

pred_logit <- predict(
  final_fit_logit_fix,
  new_data = valid_split,
  type = "prob"
) %>%
  bind_cols(valid_split %>% select(ID))

clasificacion_receta_fix <- clasificacion_receta %>%
  step_rm(ID)

final_wf_logit_fix <- final_wf %>%
  update_recipe(clasificacion_receta_fix)

final_wf_xgb_fix <- final_wf_xgb %>%
  update_recipe(clasificacion_receta_fix)
final_fit_logit_fix <- fit(
  final_wf_logit_fix,
  data = train_split
)

final_fit_xgb_fix <- fit(
  final_wf_xgb_fix,
  data = train_split
)


pred_logit <- predict(
  final_fit_logit_fix,
  new_data = valid_split,
  type = "prob"
) %>%
  bind_cols(valid_split %>% select(ID))

pred_xgb <- predict(
  final_fit_xgb_fix,
  new_data = valid_split,
  type = "prob"
) %>%
  bind_cols(valid_split %>% select(ID))

stopifnot(
  nrow(pred_logit) == nrow(valid_split),
  nrow(pred_xgb)   == nrow(valid_split)
)

any(is.na(pred_logit))
any(is.na(pred_xgb))

valid_split <- valid_split %>%
  mutate(
    Target = factor(
      Target,
      levels = c("Low", "Medium", "High")
    )
  )

train_split <- train_split %>%
  mutate(
    Target = factor(
      Target,
      levels = c("Low", "Medium", "High")
    )
  )

eval_logit <- valid_split %>%
  select(ID, Target) %>%
  inner_join(pred_logit, by = "ID")

eval_xgb <- valid_split %>%
  select(ID, Target) %>%
  inner_join(pred_xgb, by = "ID")

roc_auc(
  eval_logit,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)

roc_auc(
  eval_xgb,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)
cols <- c(".pred_Low", ".pred_Medium", ".pred_High")

pred_logit <- predict(
  final_fit_logit_fix,
  new_data = valid_split,
  type = "prob"
) %>%
  bind_cols(valid_split %>% select(ID))

pred_xgb <- predict(
  final_fit_xgb_fix,
  new_data = valid_split,
  type = "prob"
) %>%
  bind_cols(valid_split %>% select(ID))

pred_ens <- pred_logit %>%
  inner_join(pred_xgb, by = "ID", suffix = c("_logit", "_xgb")) %>%
  mutate(
    .pred_Low    = 0.4 * .pred_Low_logit    + 0.6 * .pred_Low_xgb,
    .pred_Medium = 0.4 * .pred_Medium_logit + 0.6 * .pred_Medium_xgb,
    .pred_High   = 0.4 * .pred_High_logit   + 0.6 * .pred_High_xgb
  ) %>%
  select(ID, .pred_Low, .pred_Medium, .pred_High)
eval_ens <- valid_split %>%
  select(ID, Target) %>%
  inner_join(pred_ens, by = "ID")

roc_auc(
  eval_ens,
  truth = Target,
  .pred_Low, .pred_Medium, .pred_High
)

pred_logit_sub <- predict(
  final_fit_logit_fix,
  new_data = submision,
  type = "prob"
) %>%
  bind_cols(submision %>% select(ID))

pred_xgb_sub <- predict(
  final_fit_xgb_fix,
  new_data = submision,
  type = "prob"
) %>%
  bind_cols(submision %>% select(ID))

stopifnot(
  nrow(pred_logit_sub) == nrow(submision),
  nrow(pred_xgb_sub)   == nrow(submision),
  !any(is.na(pred_logit_sub)),
  !any(is.na(pred_xgb_sub))
)

pred_ens_sub <- pred_logit_sub %>%
  inner_join(
    pred_xgb_sub,
    by = "ID",
    suffix = c("_logit", "_xgb")
  ) %>%
  mutate(
    .pred_Low    = w_logit * .pred_Low_logit    + w_xgb * .pred_Low_xgb,
    .pred_Medium = w_logit * .pred_Medium_logit + w_xgb * .pred_Medium_xgb,
    .pred_High   = w_logit * .pred_High_logit   + w_xgb * .pred_High_xgb
  ) %>%
  select(ID, .pred_Low, .pred_Medium, .pred_High)
eps <- 1e-15
to_logits <- function(p) log(p + eps)

softmax_T <- function(z1, z2, z3, T) {
  e1 <- exp(z1 / T[1])
  e2 <- exp(z2 / T[2])
  e3 <- exp(z3 / T[3])
  s  <- e1 + e2 + e3
  cbind(e1/s, e2/s, e3/s)
}

logits_sub <- pred_ens_sub %>%
  mutate(
    z_low  = to_logits(.pred_Low),
    z_med  = to_logits(.pred_Medium),
    z_high = to_logits(.pred_High)
  )

p_cal_sub <- softmax_T(
  logits_sub$z_low,
  logits_sub$z_med,
  logits_sub$z_high,
  T_opt
)

submit_df <- tibble(
  ID = logits_sub$ID,
  Low    = p_cal_sub[,1],
  Medium = p_cal_sub[,2],
  High   = p_cal_sub[,3]
)
submit_final <- pred_ens_sub %>%
  mutate(
    Target = c("Low", "Medium", "High")[
      max.col(select(., .pred_Low, .pred_Medium, .pred_High), ties.method = "first")
    ]
  ) %>%select(ID, Target)

write.csv(submit_final,'submit_stack.csv',row.names = F)

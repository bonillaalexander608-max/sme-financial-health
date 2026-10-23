library(readr)
Train <- read_csv("Train.csv")
table(Train$Target)
colSums(is.na(Train))
colnames(Train)
library(tidymodels)
library(tidyverse)

set.seed(2026)

split <- initial_split(
  Train,
  prop = 0.8,
  strata = Target
)

train_split <- training(split)
valid_split <- testing(split)
glimpse(train_split)
map_df(train_split, ~ class(.x))
prob_train<-train_split%>%
  count(country, Target) %>%
  group_by(country) %>%
  mutate(p = n / sum(n))

prob_test<-valid_split%>%
  count(country, Target) %>%
  group_by(country) %>%
  mutate(p = n / sum(n))

train_split <- train_split %>%
  left_join(prob_train %>% select(country, Target, p), by = c("country","Target")) %>%
  mutate(weight = 1 / p)

valid_split <- valid_split %>%
  left_join(prob_test %>% select(country, Target, p), by = c("country","Target")) %>%
  mutate(weight = 1 / p)
train_split<-train_split%>%dplyr::select(-p,-weight)
valid_split<-valid_split%>%dplyr::select(-p,-weight)
p_train<-train_split$weight
p_test<-valid_split$weight
rec_fhi_1 <- recipe(Target ~ ., data = train_split) %>%
  
  # 1. Roles
  update_role(ID, new_role = "id") %>%
  
  # 2. Convertir strings
  step_string2factor(all_nominal_predictors()) %>%
  
  # 3. NA explícitos en categóricas
  step_unknown(all_nominal_predictors()) %>%
  
  # 4. Imputación numérica
  step_impute_median(all_numeric_predictors()) %>%
  
  # 5. Rare levels (importantísimo)
  step_other(all_nominal_predictors(), threshold = 0.01)

prep_rec_1 <- prep(rec_fhi_1)
train_processed <- bake(prep_rec_1, new_data = train_split)
test_processed<-bake(prep_rec_1,new_data = valid_split)


#write.csv(train_processed,"train_processed.csv",row.names = F)
#write.csv(test_processed,"test_processed.csv",row.names = F)
#table(train_processed$attitude_more_successful_next_year)
#table(train_split$attitude_more_successful_next_year)
##################################################################

table(train_processed$covid_essential_service)

train_processed <- train_processed%>%
  mutate(across(starts_with("attitude_"), 
                ~ ifelse(.x == "Yes", "Yes", "No")))
train_processed<-train_processed%>%mutate(across(starts_with("perception_"),
                                  ~case_when(.x=="Yes"~"Yes",
                                             .x=="No"~"No",
                                             TRUE ~"Unknown")))
train_processed <- train_processed %>%
  mutate(across(
    starts_with("has_"),
    ~ case_when(
      .x %in% c("Have now", "Yes") ~ "Yes",
      .x %in% c("Used to have but don't have now", "Used to have but don’t have now") ~ "Used_to",
      .x %in% c("Never had", "No") ~ "No",
      TRUE ~ "Unknown"
    )
  ))
train_processed <- train_processed %>%
  mutate(
    motor_vehicle_insurance = case_when(
      motor_vehicle_insurance %in% c("Have now", "Yes") ~ "Yes",
      motor_vehicle_insurance %in% c("Used to have but don't have now", "Used to have but don’t have now") ~ "Used_to",
      motor_vehicle_insurance %in% c("Never had", "No") ~ "No",
      TRUE ~ "Unknown"
    )
  )
train_processed <- train_processed %>%
  mutate(
    compliance_income_tax = case_when(
      compliance_income_tax == "Yes" ~ "Yes",
      compliance_income_tax == "No"  ~ "No",
      TRUE                           ~ "Unknown"
    )
  )
train_processed <- train_processed %>%
  mutate(
    covid_essential_service = case_when(
      covid_essential_service == "Yes" ~ "Yes",
      covid_essential_service == "No"  ~ "No",
      TRUE                           ~ "Unknown"
    )
  )
train_processed <- train_processed %>%
  mutate(
    offers_credit_to_customers = case_when(
      offers_credit_to_customers == "Yes, always"    ~ "Always",
      offers_credit_to_customers == "Yes, sometimes" ~ "Sometimes",
      offers_credit_to_customers == "No"             ~ "No",
      TRUE                                           ~ "Unknown"
    )
  )
train_processed <- train_processed %>%
  mutate(
    keeps_financial_records = case_when(
      keeps_financial_records == "Yes, always"    ~ "Always",
      keeps_financial_records == "Yes, sometimes" ~ "Sometimes",
      keeps_financial_records == "No"             ~ "No",
      TRUE                                           ~ "Unknown"
    )
  )
train_processed <- train_processed %>%
  mutate(across(
    c(uses_friends_family_savings, uses_informal_lender),
    ~ case_when(
      .x == "Have now" ~ "Yes",
      .x %in% c("Used to have but don't have now", "Used to have but don’t have now") ~ "Used_to",
      .x == "Never had" ~ "No",
      TRUE ~ "Unknown"
    )
  ))
train_processed <- train_processed %>%
  mutate(across(
    c(medical_insurance, funeral_insurance),
    ~ case_when(
      .x == "Have now" ~ "Yes",
      .x %in% c("Used to have but don't have now", "Used to have but don’t have now") ~ "Used_to",
      .x == "Never had" ~ "No",
      TRUE ~ "Unknown"
    )
  ))
train_processed <- train_processed %>%
  mutate(
    owner_sex = case_when(
      owner_sex == "Male"   ~ "Male",
      owner_sex == "Female" ~ "Female",
      TRUE                  ~ "Unknown"
    )
  )
##funcion
limpiar_datos_empresa <- function(df) {
  df_limpio <- df %>%
    # 1. Variables de Actitud (Binarias: Yes vs Resto)
    mutate(across(starts_with("attitude_"), 
                  ~ ifelse(.x == "Yes", "Yes", "No"))) %>%
    
    # 2. Variables de Percepción y Cumplimiento (Yes, No, Unknown)
    mutate(across(c(starts_with("perception_"), 
                    compliance_income_tax, 
                    covid_essential_service),
                  ~ case_when(
                    .x == "Yes" ~ "Yes",
                    .x == "No"  ~ "No",
                    TRUE        ~ "Unknown"))) %>%
    
    # 3. Variables de Posesión y Uso (Yes, No, Used_to, Unknown)
    mutate(across(c(starts_with("has_"), 
                    starts_with("uses_"),
                    motor_vehicle_insurance, 
                    medical_insurance, 
                    funeral_insurance),
                  ~ case_when(
                    .x %in% c("Have now", "Yes") ~ "Yes",
                    .x %in% c("Used to have but don't have now", 
                              "Used to have but don’t have now") ~ "Used_to",
                    .x %in% c("Never had", "No") ~ "No",
                    TRUE ~ "Unknown"))) %>%
    
    # 4. Variables de Frecuencia (Always, Sometimes, No, Unknown)
    mutate(across(c(offers_credit_to_customers, keeps_financial_records),
                  ~ case_when(
                    .x == "Yes, always"    ~ "Always",
                    .x == "Yes, sometimes" ~ "Sometimes",
                    .x == "No"             ~ "No",
                    TRUE                   ~ "Unknown"))) %>%
    
    # 5. Sexo del dueño
    mutate(owner_sex = case_when(
      owner_sex == "Male"   ~ "Male",
      owner_sex == "Female" ~ "Female",
      TRUE                  ~ "Unknown"
    ))
  
  return(df_limpio)
}

train_processed<-limpiar_datos_empresa(train_processed)
test_processed <- limpiar_datos_empresa(test_processed)


train_final <- train_final %>%
  mutate(Target = case_when(
    Target == "Low"    ~ 0,
    Target == "Medium" ~ 1,
    Target == "High"   ~ 2
  ))
test_final <- test_final %>%
  mutate(Target = case_when(
    Target == "Low"    ~ 0,
    Target == "Medium" ~ 1,
    Target == "High"   ~ 2
  ))


submision<-bake(prep_rec_1,new_data = submision)
submision<-limpiar_datos_empresa(submision)

#########analisis exploratorio 
library(tidyverse)

train_processed %>%
  group_by(Target) %>%
  summarise(
    n = n(),
    turnover_median = median(business_turnover, na.rm = TRUE),
    turnover_mean   = mean(business_turnover, na.rm = TRUE),
    expenses_median = median(business_expenses, na.rm = TRUE),
    income_median   = median(personal_income, na.rm = TRUE)
  )
ggplot(train_processed, aes(x = Target, y = business_turnover)) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Business Turnover by Target",
    y = "Business Turnover"
  ) +
  theme_minimal()
ggplot(train_processed, aes(x = Target, y = business_expenses)) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Business Expenses by Target",
    y = "Business Expenses"
  ) +
  theme_minimal()
ggplot(train_processed, aes(x = Target, y = personal_income)) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Personal Income by Target",
    y = "Personal Income"
  ) +
  theme_minimal()
train_processed %>%
  select(business_turnover, business_expenses, personal_income) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50) +
  facet_wrap(~ name, scales = "free") +
  theme_minimal()
train_processed %>%
  mutate(
    log_turnover = log1p(business_turnover),
    log_expenses = log1p(business_expenses),
    log_income   = log1p(personal_income)
  ) %>%
  select(Target, starts_with("log_")) %>%
  pivot_longer(-Target) %>%
  ggplot(aes(x = Target, y = value)) +
  geom_boxplot(outlier.alpha = 0.2) +
  facet_wrap(~ name, scales = "free_y") +
  theme_minimal()
ggplot(train_processed,
       aes(x = business_turnover, y = business_expenses, color = Target)) +
  geom_point(alpha = 0.3) +
  scale_x_log10() +
  scale_y_log10() +
  theme_minimal()
library(car)
yj_modelo <- powerTransform(train_processed$personal_income, family="yjPower")
yj_modelo$lambda
#logaritmar las variables 
###################################################
train_processed %>%
  summarise(
    cor_age = cor(business_age_years, business_age_months, use = "complete.obs")
  )
ggplot(train_processed, aes(x = Target, y = business_age_years)) +
  geom_boxplot(outlier.alpha = 0.2) +
  theme_minimal() +
  labs(
    title = "Business Age (Years) by Target",
    y = "Years"
  )
ggplot(train_processed, aes(x = Target, y = business_age_months)) +
  geom_boxplot(outlier.alpha = 0.2) +
  theme_minimal() +
  labs(
    title = "Business Age (Months) by Target",
    y = "Months"
  )
ggplot(train_processed, aes(x = Target, y = owner_age)) +
  geom_boxplot(outlier.alpha = 0.2) +
  theme_minimal() +
  labs(
    title = "Owner Age by Target",
    y = "Owner Age"
  )
train_processed %>%
  select(Target, business_age_years, owner_age) %>%
  pivot_longer(-Target) %>%
  ggplot(aes(x = value, fill = Target)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ name, scales = "free") +
  theme_minimal()
ggplot(train_processed,
       aes(x = owner_age, y = business_age_years, color = Target)) +
  geom_point(alpha = 0.3) +
  theme_minimal()
########senal debill de las 3 variable 
###################################
prop_by_target <- function(df, var) {
  df %>%
    count(Target, !!sym(var)) %>%
    group_by(Target) %>%
    mutate(p = n / sum(n))
}
prop_by_target(train_processed, "keeps_financial_records")
prop_by_target(train_processed, "has_debit_card")
prop_by_target(train_processed, "has_credit_card")
prop_by_target(train_processed, "has_loan_account")
prop_by_target(train_processed, "has_insurance")
prop_by_target(train_processed, "has_mobile_money")
prop_by_target(train_processed, "has_internet_banking")
#Mantén: keeps_financial_records (ordinal),has_credit_card (binaria)
#has_loan_account (binaria),has_debit_card (binaria)
#Usa:Unknown como categoría informativa
####################################
insurance_vars <- c(
  "motor_vehicle_insurance",
  "medical_insurance",
  "funeral_insurance",
  "perception_insurance_important",
  "perception_cannot_afford_insurance",
  "perception_insurance_doesnt_cover_losses",
  "perception_insurance_companies_dont_insure_businesses_like_yours"
)
prop_by_target <- function(df, var){
  df %>%
    count(Target, .data[[var]]) %>%
    group_by(Target) %>%
    mutate(p = n / sum(n)) %>%
    ungroup()
}
insurance_tables <- lapply(
  insurance_vars,
  function(v) prop_by_target(train_processed, v)
)

names(insurance_tables) <- insurance_vars
chisq_results <- lapply(
  insurance_vars,
  function(v){
    tbl <- table(train_processed[[v]], train_processed$Target)
    chisq.test(tbl)
  }
)

names(chisq_results) <- insurance_vars
chisq_results$medical_insurance
chisq_results$perception_cannot_afford_insurance
chisq_results$motor_vehicle_insurance
chisq_results
cramers_v <- function(x, y){
  tbl <- table(x, y)
  chi2 <- suppressWarnings(chisq.test(tbl)$statistic)
  n <- sum(tbl)
  k <- min(dim(tbl))
  sqrt(chi2 / (n * (k - 1)))
}
cramer_results <- tibble(
  variable = insurance_vars,
  cramer_v = sapply(
    insurance_vars,
    function(v) cramers_v(train_processed[[v]], train_processed$Target)
  )
)
cramer_results %>%
  mutate(
    strength = case_when(
      cramer_v < 0.1 ~ "Very weak",
      cramer_v < 0.2 ~ "Weak",
      cramer_v < 0.4 ~ "Moderate",
      cramer_v < 0.6 ~ "Strong",
      TRUE ~ "Very strong"
    )
  ) %>%
  arrange(desc(cramer_v))

#funeral, medical, motor van al modelo si o si
insurance_perception_index <- train_processed %>%
  transmute(
    cannot_afford = ifelse(perception_cannot_afford_insurance == "Yes", 1, 0),
    doesnt_cover  = ifelse(perception_insurance_doesnt_cover_losses == "Yes", 1, 0),
    excluded      = ifelse(
      perception_insurance_companies_dont_insure_businesses_like_yours == "Yes", 1, 0
    ),
    not_important = ifelse(perception_insurance_important == "No", 1, 0)
  )
train_processed$insurance_perception_index <-
  rowSums(insurance_perception_index/4, na.rm = TRUE)

train_processed %>%
  group_by(Target) %>%
  summarise(
    mean_index = mean(insurance_perception_index),
    median_index = median(insurance_perception_index)
  )
cramers_v(train_processed$insurance_perception_index,
          train_processed$Target)
train_processed<-train_processed%>%select(-insurance_perception_index)
############################################################################
attitude_vars <- c(
  "attitude_stable_business_environment",
  "attitude_worried_shutdown",
  "attitude_satisfied_with_achievement",
  "attitude_more_successful_next_year",
  "motivation_make_more_money"
)

lapply(attitude_vars, function(v) {
  prop_by_target(train_processed, v)
})
stress_vars <- c(
  "problem_sourcing_money",
  "current_problem_cash_flow"
)

lapply(stress_vars, function(v) {
  prop_by_target(train_processed, v)
})
informal_vars <- c(
  "uses_friends_family_savings",
  "uses_informal_lender",
  "offers_credit_to_customers"
)

lapply(informal_vars, function(v) {
  prop_by_target(train_processed, v)
})
context_vars <- c(
  "covid_essential_service",
  "marketing_word_of_mouth",
  "future_risk_theft_stock",
  "compliance_income_tax"
)

lapply(context_vars, function(v) {
  prop_by_target(train_processed, v)
})
library(rstatix)

library(rstatix)
library(tibble)
library(dplyr)

cramer_results_2 <- lapply(
  c(attitude_vars, stress_vars, informal_vars, context_vars),
  function(v) {
    tibble(
      variable = v,
      cramer_v = cramers_v(
        train_processed[[v]],
        train_processed$Target
      )
    )
  }
)

cramer_results_2 <- bind_rows(cramer_results_2) %>%
  arrange(desc(cramer_v))

cramer_results_2
table(train_processed$owner_sex,train_processed$Target)
prop.table(
  table(train_processed$owner_sex, train_processed$Target),
  margin = 1
)
cramers_v(train_processed$owner_sex, train_processed$Target)

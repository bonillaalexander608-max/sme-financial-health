suppressPackageStartupMessages({library(dplyr);library(forcats)})
# Preserved from the supplied xgb_sin_ID.R; used before the fitted recipe.
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

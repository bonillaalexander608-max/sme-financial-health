clasificacion_receta<-recipe(Target~.,data = train_processed)%>%
  update_role(ID,country,new_role = "id")%>%
  step_log(personal_income,business_turnover,business_expenses,offset = 1)%>%
  step_mutate(across(starts_with("has_"),~ifelse(.x=="Yes",1,0)))%>%
  step_rm(starts_with("attitude_"),starts_with("business_age"),starts_with("perception_"),,problem_sourcing_money,
          covid_essential_service,owner_age)%>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%  
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

receta_prep <- prep(
  clasificacion_receta,
  training = train_processed,
  verbose = TRUE
)
juice(receta_prep)
glimpse(juice(receta_prep))

train_enet<-bake(receta_prep,new_data = train_processed)
test_enet<-bake(receta_prep,new_data = test_processed)
submision_enet<-bake(receta_prep,new_data = submision)
train_enet<-train_enet%>%mutate(peso=p_train)
test_enet<-test_enet%>%mutate(peso=p_test)
total_enet<-rbind(train_enet,test_enet)
write.csv(train_enet,"train_enet.csv",row.names = F)
write.csv(test_enet,"test_enet.csv",row.names = F)
write.csv(submision_enet,"submision_enet.csv",row.names = F)
write.csv(total_enet,"total_enet.csv",row.names = F)

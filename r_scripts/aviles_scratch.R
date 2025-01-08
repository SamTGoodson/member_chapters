suppressPackageStartupMessages(library(tidyverse))
library(rio)
suppressPackageStartupMessages(library(flextable))
library(gt)
library(RColorBrewer)
library(sf)
library(stargazer)
library(modelsummary)
library(fixest)
library(tidymodels)
library(vip)

meta_clusters <- read_csv('../chapter_two/data/meta_clusters_with_formatted_name.csv')
bsr <- read_csv('../cc_legislation/data/bills_with_cluster_count.csv')
c3_results <- read_csv("../cc_election_cleaning/election_results_with_vote_sponsor_cluster_DEC04.csv")
demo_clusters <- read_csv("../cc_election_cleaning/district_level_demo_clusters.csv")
ed_sf <- read_sf('../neighborhoods_ccdistricts/data/shapefiles/ed')


aa <- c3_results %>%
  filter(candidate == 'Alexa Aviles')
yl <- c3_results %>%
  filter(candidate == 'Yu Lin')
top_two <- c3_results %>%
  filter(candidate == 'Alexa Aviles' | candidate == 'Yu Lin')

top_two$ElectDist <- top_two$ElectDist_x
tt_map <- left_join(top_two,ed_sf, by = 'ElectDist')

top_two$geometry <- st_as_sfc(top_two$geometry)
top_two <- st_as_sf(top_two)
top_two%>%
  drop_na(geometry,vote_share)%>%
ggplot() +
  geom_sf(aes(fill = vote_share,geometry = geometry)) + 
  theme_minimal()

top_two <- top_two %>%
  group_by(ElectDist) %>%
  mutate(winner = if_else(vote_share == max(vote_share), candidate, NA_character_)) %>%
  ungroup()
top_two_max <- top_two %>%
  filter()

ggplot(top_two) +
  geom_sf(aes(fill = winner, geometry = geometry, alpha = vote_share)) +
  scale_fill_manual(
    values = c("Yu Lin" = "darkblue", "Alexa Aviles" = "darkred"),
    na.value = "gray80"
  ) +
  scale_alpha(range = c(0.5, 1)) +  # Adjust transparency for vote share
  labs(fill = "Winning Candidate") +
  theme_minimal()

lm1 <- lm(vote_share ~ log(mhhi21) + nhb21p, data = aa)
lm2 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p  , data = aa)
lm3 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio, data = aa)
lm4 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp, data = aa)
lm4 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp + hh21op  + summer_noise_complaints + mean_noise , data = aa)
lm5 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp + hh21op + summer_noise_complaints + mean_noise  + perc_retail , data = aa)
stargazer(lm1,lm2,lm3,lm4,lm5, type = 'html',out = 'models.html', covariate.labels=c('Log MHHI', 'NH Black Share','Hispanic Share','White Transplant Share',
                                                                 '% BA or more','Homeowner Share','Summer 2020 Noise','Avg. Noise',
                                                                 '% Retail Employees'),omit.stat=c("LL","ser","f"))

colnames(aa)
modeling <- aa%>%
  select(!c(Precinct:total_vote_precinct,ed_name:geometry))
cor_matrix <- cor(modeling, use = "complete.obs")
cor_target <- cor_matrix["vote_share", ]
# Sort by strongest correlations
cor_target[order(abs(cor_target), decreasing = TRUE)]

# random forest for fun
set.seed(123)
splits      <- initial_split(aa, strata = vote_share)
aa_other <- training(splits)
aa_test  <- testing(splits)

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression")
rf_recipe <- 
  recipe(vote_share ~ ., data = aa)
rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod)%>%
  add_recipe(rf_recipe)
val_set <- validation_split(aa_other, 
                            strata = vote_share, 
                            prop = 0.80)

rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


winners <- c3_results%>%
  filter(winner == TRUE)

winners <- winners %>%
  mutate(is_three = if_else(kmode_cluster == 3, 1, 0, missing = 0))

winners%>%
  select(ElectDist_x,candidate,kmode_cluster,is_three)%>%
  view()

winners_3 <- winners%>%
  filter(district_cluster == 3)
modeling <- winners_3%>%
  select(!c(Precinct:total_vote_precinct,ed_name:geometry))
str(modeling$is_three)
levels(modeling$is_three)
modeling$is_three <- as.factor(modeling$is_three)

set.seed(123)
splits      <- initial_split(modeling, strata = is_three)
modeling_other <- training(splits)
modelnig_test  <- testing(splits)
val_set <- validation_split(modeling_other, 
                            strata = is_three, 
                            prop = 0.80)

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")
rf_recipe <- 
  recipe(is_three ~ ., data = modeling_other)%>%
  step_rm(vote_share,garcia213p,dpp20bs,adams213p)
rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))
rf_res %>% 
  show_best(metric = "roc_auc")

rf_best <- 
  rf_res %>% 
  select_best(metric = "roc_auc")
rf_best
rf_res %>% 
  collect_predictions()

rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(children, .pred_is_three) %>% 
  mutate(model = "Random Forest")


last_rf_mod <- 
  rand_forest(mtry = 32, min_n = 37, trees = 1000) %>% 
  set_engine("ranger",  importance = "impurity") %>% 
  set_mode("classification")

# the last workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

# the last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)


last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)
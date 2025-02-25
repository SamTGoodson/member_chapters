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
library(leaflet)
getwd()
meta_clusters <- read_csv('../chapter_two/data/meta_clusters_with_formatted_name.csv')
bsr <- read_csv('../cc_legislation/data/bills_with_cluster_count.csv')
c3_results <- read_csv("../cc_election_cleaning/election_results_with_vote_sponsor_cluster_DEC04.csv")
demo_clusters <- read_csv("../cc_election_cleaning/district_level_demo_clusters.csv")
ed_sf <- read_sf('../neighborhoods_ccdistricts/data/shapefiles/ed')

tc <- c3_results %>%
  filter(candidate == 'Tiffany Caban')
eh <- c3_results %>%
  filter(candidate == 'Evie Hantzopoulos')
top_two <- c3_results %>%
  filter(candidate == 'Tiffany Caban' | candidate == 'Evie Hantzopoulos')
top_two$ElectDist <- top_two$ElectDist_x
top_two <- top_two %>%
  group_by(ElectDist) %>%
  mutate(winner = if_else(vote_share == max(vote_share), candidate, NA_character_)) %>%
  ungroup()
top_two_max <- top_two %>%
  filter()

lm1 <- lm(vote_share ~ log(mhhi21) + nhb21p, data = tc)
lm2 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p  , data = tc)
lm3 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio, data = tc)
lm4 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp, data = tc)
lm4 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp + hh21op  + summer_noise_complaints + mean_noise , data = tc)
lm5 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp + hh21op + summer_noise_complaints + mean_noise  + perc_retail , data = tc)
stargazer(lm1,lm2,lm3,lm4,lm5, type = 'html',out = 'models.html', covariate.labels=c('Log MHHI', 'NH Black Share','Hispanic Share','White Transplant Share',
                                                                                     '% BA or more','Homeowner Share','Summer 2020 Noise','Avg. Noise',
  '% Retail Employees'),omit.stat=c("LL","ser","f"))

cor_table <- tc %>%
  select(vote_share,mhhi21,  nhb21p, h21p, white_transplant_ratio, cvap21bapp,
        hh21op, summer_noise_complaints, mean_noise , perc_retail,bus_ratio,wfh_ratio,
        adams213p,dpp20bs)%>%
  cor(use = "pairwise.complete.obs")
print(cor_table)

# Load necessary libraries
library(dplyr)

# Scaled regressions
tc_scaled <- tc %>%
  mutate(across(c(mhhi21, nhb21p, h21p, white_transplant_ratio, cvap21bapp, 
                  hh21op, summer_noise_complaints, mean_noise, perc_retail), scale))

lm1 <- lm(vote_share ~ log(mhhi21) + nhb21p, data = tc_scaled)
lm2 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p, data = tc_scaled)
lm3 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio, data = tc_scaled)
lm4 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp, data = tc_scaled)
lm5 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp +
            hh21op, data = tc_scaled)
lm6 <- lm(vote_share ~ log(mhhi21) + nhb21p + h21p + white_transplant_ratio + cvap21bapp +
            hh21op + summer_noise_complaints + mean_noise + perc_retail, data = tc_scaled)
stargazer(lm1, lm2, lm3, lm4, lm5,lm6, 
          type = 'html', 
          out = 'models.html', 
          covariate.labels = c('Log MHHI', 'NH Black Share', 'Hispanic Share', 
                               'White Transplant Share', '% BA or more', 'Homeowner Share', 
                               'Summer 2020 Noise', 'Avg. Noise', '% Retail Employees'),
          omit.stat = c("LL", "ser", "f"))

vif(lm6)
# compare won districts 
top_two%>%
  filter(rank == 1)%>%
  group_by(candidate)%>%
  summarise(
    wtr = mean(white_transplant_ratio),
    hisp = mean(h21p),
    ba = mean(cvap21bapp),
    ho = mean(hh21op),
    income = mean(mhhi21),
    retail = mean(perc_retail),
    health = mean(perc_healthcare),
    wfh = mean(wfh_ratio),
    bus = mean(bus_ratio),
    drive = mean(drive_ratio),
    dense = mean(ldensity),
    bernie = mean(dpp20bs)
  )

compare <- top_two %>%
  filter(rank == 1)

t.test(dpp20bs ~ candidate, data = compare, var.equal = TRUE)
top_two <- st_as_sf(top_two,geometry = geometry)
top_two <- st_as_sf(top_two, wkt = "geometry", crs = 2263)
st_geometry(top_two)
compare <- top_two %>%
  filter(rank == 1)%>%
  group_by(Precinct,candidate) %>%
  summarise(
    vote_share = mean(vote_share, na.rm = TRUE),
    geometry = st_union(geometry),
    wtr = mean(white_transplant_ratio),
    hisp = mean(h21p),
    ba = mean(cvap21bapp),
    ho = mean(hh21op),
    income = mean(mhhi21),
    retail = mean(perc_retail),
    health = mean(perc_healthcare),
    wfh = mean(wfh_ratio),
    bus = mean(bus_ratio),
    drive = mean(drive_ratio),
    dense = mean(ldensity),
    bernie = mean(dpp20bs),
    garcia213p = mean(garcia213p),
    adams213p = mean(adams213p),
    nhw21p = mean(nhw21p),
    mean_noise = mean(mean_noise)
  )

compare <- st_transform(compare, crs = 4326)
pal <- colorNumeric(palette = "viridis", 
                     domain = compare$ho)


eh <- st_as_sf(eh, wkt = "geometry", crs = 2263)
compare <- st_transform(eh, crs = 4326)
pal <- colorNumeric(palette = "viridis", 
                    domain = compare$vote_share)

leaflet(compare)%>%
  addProviderTiles(provider = "CartoDB.Positron") %>%
  addPolygons(popup = ~paste0(
                              round(vote_share,2),'% for',candidate,'</br>',
                              round(nhw21p,2),'% White','</br>',
                              round(white_transplant_ratio,2),'% White Transplants','</br>',
                              '$',round(mhhi21,2),'Median Household Income','</br>',
                              round(cvap21bapp,2),'% ba or more','</br>',
                              'Average Noise Complaints',round(mean_noise,2),'</br>',
                              round(adams213p,2),'% for Adams','</br>',
                              round(garcia213p,2),'% for Garcia','</br>',
                              round(hh21op,2),'% Home Ownership'
                              ),
                          
              stroke = FALSE,
              smoothFactor = 0,
              fillOpacity = 0.7,
              color = ~ pal(vote_share))
library(tidyverse)

df = read_csv2('Data/ERCompanyLevel.csv')

p1 <- df %>% 
  ggplot() + 
  #group_by(Sector) %>%
  geom_bar(mapping = aes(x = Jaar, y = Emissie, fill = Sector), stat = 'identity') +
  labs(title = "Emission per sector per jaar") +
  theme_hw()

p1
print(p1)
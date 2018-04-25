library(tidyverse)
library(gridExtra)

setwd("/Users/conormcdonald/Desktop/pytorch_blogpost1")
decreg <- read_csv("data/decision_region.csv", col_names = c("X1", "X2", "yhat"))
moons <- read_csv("data/moons_data.csv")
loss_acc <- read_csv("data/loss_acc.csv")

dec_reg_plot <- decreg %>%
ggplot(aes(X1, X2, colour = factor(yhat))) +
  geom_point(alpha = .4, size = 4) +
  geom_point(data = moons, aes(X1, X2), colour = "dodger blue", alpha = .7) +
  theme_minimal() +
  guides(colour = FALSE) + 
  labs(title = "Model Decision regions")

loss_acc$ix <- seq(nrow(loss_acc))

p1 <- loss_acc %>% 
  ggplot(aes(ix, loss)) + 
  geom_line(aes(colour = loss), alpha = .5, size= .8) +
  theme_minimal() +
  labs(title = "Loss (BCE) across model iterations") + 
  guides(colour = FALSE)

p2 <- loss_acc %>% 
  ggplot(aes(ix, accuracy)) + 
  geom_line(aes(colour = loss), alpha = .5, size= .8) +
  theme_minimal() +
  labs(title = "Accuracy across model iterations") + 
  guides(colour = FALSE)

p3 <- gridExtra::grid.arrange(p1, p2, ncol = 1)

list.dirs()
grepl("plots", "plots in")

dir.create("plots")

ggsave("plots/loss_acc.png", p3)
ggsave("plots/dec_reg.png", dec_reg_plot)

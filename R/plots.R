library(tidyverse)
library(gridExtra)

setwd("/Users/conormcdonald/Desktop/pytorch_blogpost1")
decreg <- read_csv("data/decision_region.csv", col_names = c("X1", "X2", "yhat"))
moons <- read_csv("data/moons_data.csv", col_names = c("X1", "X2", "yhat"))
loss_acc <- read_csv("data/model_performance.csv")

dec_reg_plot <- decreg %>%
  ggplot(aes(X1, X2, colour = factor(yhat))) +
  geom_point(alpha = .2, size = 4) +
  geom_point(data = moons, aes(X1, X2), colour = "dodger blue", alpha = .7) +
  theme_minimal() +
  guides(colour = FALSE) +
  labs(title = "Model Decision regions")

dec_reg_plot

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

p4 <- loss_acc %>%
  ggplot(aes(X1, losses)) + 
  geom_point(alpha=.3) + 
  geom_line(colour="dodger blue", alpha = .7) + 
  theme_minimal() + 
  labs(title = "Model loss", 
       x = "Iteration", 
       y = "Binary cross entropy")

p5 <- loss_acc %>%
  ggplot(aes(X1, acc)) + 
  geom_point(alpha=.3) + 
  geom_line(colour="dodger blue", alpha = .7) + 
  theme_minimal() + 
  labs(title = "Model accuracy", 
       x = "Iteration", 
       y = "Accuracy")

p6 <- gridExtra::grid.arrange(p4, p5, ncol = 2)


list.dirs()
grepl("plots", "plots in")

dir.create("plots")

ggsave("plots/loss_acc.png", p3)
ggsave("plots/dec_reg.png", dec_reg_plot)


# create logistic regression classifier and decision region

clf_lr <- glm(y ~ ., data = moons, family = binomial(link='logit'))
clf_rf <- ranger(y ~ ., data = moons_data)

grid <- as_data_frame(expand.grid(
  x1 = seq(min(moons_data$x1), max(moons_data$x1), .05),
  x2 = seq(min(moons_data$x2), max(moons_data$x2), .05)
))

grid_preds_lr <- as.numeric(predict.glm(clf_lr, grid, type = "response"))
grid_preds_rf <- predict(clf_rf, grid)$predictions

grid$preds_lr <- round(grid_preds)
grid$preds_rf <- round(grid_preds_rf)

p1 <- grid %>% 
  ggplot(aes(x1, x2, colour = factor(preds_lr))) + 
  geom_point(alpha = .2, size = 3) + 
  geom_point(data = moons_data, aes(x1, x2, colour = factor(y), alpha = .4)) + 
  theme_minimal() + 
  guides(alpha = FALSE, 
         colour = FALSE) + 
  labs(title = "Logistic regression decision boundary")

p2 <- grid %>% 
  ggplot(aes(x1, x2, colour = factor(preds_rf))) + 
  geom_point(alpha = .2, size = 3) + 
  geom_point(data = moons_data, aes(x1, x2, colour = factor(y), alpha = .4)) + 
  theme_minimal() + 
  guides(alpha = FALSE, 
         colour = FALSE) + 
  labs(title = "Random Forest regression decision boundary")

grid.arrange(p1, p2, ncol = 2)
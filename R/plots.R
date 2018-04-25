library(tidyverse)

setwd("/Users/conormcdonald/Desktop/pytorch_blogpost1")
decreg <- read_csv("data/decision_region.csv", col_names = c("X1", "X2", "yhat"))
moons <- read_csv("data/moons_data.csv")


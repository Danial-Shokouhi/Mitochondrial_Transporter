library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(car)
library(rstatix)
library(writexl)

df <- read_excel("Enzyme.xlsx", sheet = "Mito") %>%
  rename(Sample = `...1`) %>%
  mutate(Group = factor(ifelse(str_detect(Sample, "^B"), "Mutant", "WT")))

enzymes <- c("GDC", "PDH", "OGDH", "BCKDH", "SDH")

results_list <- list()

for (enz in enzymes) {

  data_sub <- df %>%
    select(Group, all_of(enz)) %>%
    rename(value = all_of(enz)) %>%
    filter(!is.na(value))

  wt_vals <- data_sub %>% filter(Group == "WT") %>% pull(value)
  mut_vals <- data_sub %>% filter(Group == "Mutant") %>% pull(value)

  # Normality check per group (skipped if n < 3)
  shapiro_wt <- if (length(wt_vals) >= 3) shapiro.test(wt_vals) else list(p.value = NA)
  shapiro_mut <- if (length(mut_vals) >= 3) shapiro.test(mut_vals) else list(p.value = NA)

  normal_flag <- ifelse(
    is.na(shapiro_wt$p.value) | is.na(shapiro_mut$p.value),
    FALSE,
    (shapiro_wt$p.value > 0.05 & shapiro_mut$p.value > 0.05)
  )

  levene <- car::leveneTest(value ~ Group, data = data_sub)
  equal_var <- levene[["Pr(>F)"]][1] > 0.05

  # t-test if both groups normal, otherwise Mann-Whitney
  if (normal_flag) {
    test <- t.test(value ~ Group, data = data_sub, var.equal = equal_var)
    test_used <- ifelse(equal_var, "Student t-test", "Welch t-test")
  } else {
    test <- wilcox.test(value ~ Group, data = data_sub, exact = FALSE)
    test_used <- "Mann-Whitney U test"
  }

  effect <- tryCatch(
    cohens_d(data_sub, value ~ Group),
    error = function(e) data.frame(effsize = NA)
  )

  summary_stats <- data_sub %>%
    group_by(Group) %>%
    summarise(mean = mean(value), sd = sd(value), n = n(), .groups = "drop")

  wt_mean <- summary_stats$mean[summary_stats$Group == "WT"]
  wt_sd <- summary_stats$sd[summary_stats$Group == "WT"]
  mut_mean <- summary_stats$mean[summary_stats$Group == "Mutant"]
  mut_sd <- summary_stats$sd[summary_stats$Group == "Mutant"]

  results_list[[enz]] <- data.frame(
    Enzyme = enz,
    Test_used = test_used,
    WT_mean = wt_mean,
    WT_sd = wt_sd,
    Mutant_mean = mut_mean,
    Mutant_sd = mut_sd,
    p_value = test$p.value,
    effect_size = effect$effsize[1],
    normality_WT = shapiro_wt$p.value,
    normality_Mutant = shapiro_mut$p.value,
    levene_p = levene[["Pr(>F)"]][1]
  )
}

stats_df <- bind_rows(results_list)
stats_df$p_adj <- p.adjust(stats_df$p_value, method = "BH")

write_xlsx(list(Statistics = stats_df), "Enzyme_Statistics_Results.xlsx")

print(stats_df)

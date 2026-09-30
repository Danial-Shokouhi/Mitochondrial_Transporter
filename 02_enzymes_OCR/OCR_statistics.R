library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(car)
library(rstatix)
library(writexl)

df_raw <- read_excel("OCR.xlsx", sheet = "OCR_data")

df_long <- df_raw %>%
  pivot_longer(cols = -Substrate, names_to = "Sample", values_to = "Value") %>%
  mutate(
    Group = factor(
      ifelse(str_detect(Sample, "^B"), "Mutant", "WT"),
      levels = c("WT", "Mutant")
    )
  ) %>%
  filter(!is.na(Value))

Substrate <- unique(df_long$Substrate)

results_list <- list()

for (met in Substrate) {

  data_sub <- df_long %>% filter(Substrate == met)

  wt_vals <- data_sub %>% filter(Group == "WT") %>% pull(Value)
  mut_vals <- data_sub %>% filter(Group == "Mutant") %>% pull(Value)

  # Normality check per group (skipped if n < 3)
  shapiro_wt <- if (length(wt_vals) >= 3) shapiro.test(wt_vals) else list(p.value = NA)
  shapiro_mut <- if (length(mut_vals) >= 3) shapiro.test(mut_vals) else list(p.value = NA)

  normal_flag <- ifelse(
    is.na(shapiro_wt$p.value) | is.na(shapiro_mut$p.value),
    FALSE,
    (shapiro_wt$p.value > 0.05 & shapiro_mut$p.value > 0.05)
  )

  levene <- car::leveneTest(Value ~ Group, data = data_sub)
  equal_var <- levene[["Pr(>F)"]][1] > 0.05

  # t-test if both groups normal, otherwise Mann-Whitney
  if (normal_flag) {
    test <- t.test(Value ~ Group, data = data_sub, var.equal = equal_var)
    test_used <- ifelse(equal_var, "Student t-test", "Welch t-test")
  } else {
    test <- wilcox.test(Value ~ Group, data = data_sub, exact = FALSE)
    test_used <- "Mann-Whitney U test"
  }

  effect <- tryCatch(
    cohens_d(data_sub, Value ~ Group),
    error = function(e) data.frame(effsize = NA)
  )

  summary_stats <- data_sub %>%
    group_by(Group) %>%
    summarise(
      mean = mean(Value, na.rm = TRUE),
      sd = sd(Value, na.rm = TRUE),
      median = median(Value, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )

  wt_mean <- summary_stats$mean[summary_stats$Group == "WT"]
  wt_sd <- summary_stats$sd[summary_stats$Group == "WT"]
  wt_median <- summary_stats$median[summary_stats$Group == "WT"]

  mut_mean <- summary_stats$mean[summary_stats$Group == "Mutant"]
  mut_sd <- summary_stats$sd[summary_stats$Group == "Mutant"]
  mut_median <- summary_stats$median[summary_stats$Group == "Mutant"]

  results_list[[met]] <- data.frame(
    Metabolite = met,
    Test_used = test_used,
    WT_mean = wt_mean,
    WT_sd = wt_sd,
    WT_median = wt_median,
    Mutant_mean = mut_mean,
    Mutant_sd = mut_sd,
    Mutant_median = mut_median,
    p_value = test$p.value,
    Effect_size_Cohens_d = effect$effsize[1],
    Shapiro_WT_p = shapiro_wt$p.value,
    Shapiro_Mutant_p = shapiro_mut$p.value,
    Levene_p = levene[["Pr(>F)"]][1]
  )
}

stats_df <- bind_rows(results_list)
stats_df$p_adj_BH <- p.adjust(stats_df$p_value, method = "BH")

stats_df <- stats_df %>%
  mutate(
    Significance = case_when(
      p_adj_BH < 0.0001 ~ "****",
      p_adj_BH < 0.001  ~ "***",
      p_adj_BH < 0.01   ~ "**",
      p_adj_BH < 0.05   ~ "*",
      TRUE ~ "ns"
    )
  )

write_xlsx(list(Statistics = stats_df), "stats_OCR.xlsx")

print(stats_df)

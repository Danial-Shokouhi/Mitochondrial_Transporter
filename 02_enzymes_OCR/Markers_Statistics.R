library(readxl)
library(dplyr)
library(tidyr)
library(emmeans)
library(writexl)

df <- read_excel("Marker.xlsx", sheet = "analysis")
names(df) <- trimws(names(df))
print(names(df))

df <- df %>%
  mutate(
    Fraction = trimws(as.character(Fraction)),
    Genotype = trimws(as.character(Genotype)),
    CS = as.numeric(CS),
    HPR = as.numeric(HPR),
    UGPase = as.numeric(UGPase),
    Chlorophyll = as.numeric(Chlorophyll)
  ) %>%
  filter(
    Fraction %in% c("crude", "Mito"),
    Genotype %in% c("Col-0", "bou")
  )

df$Fraction <- factor(df$Fraction, levels = c("crude", "Mito"))
df$Genotype <- factor(df$Genotype, levels = c("Col-0", "bou"))

markers <- c("CS", "HPR", "UGPase", "Chlorophyll")

analyse_marker <- function(marker) {

  dat <- df %>%
    select(Fraction, Genotype, all_of(marker)) %>%
    rename(Value = all_of(marker)) %>%
    filter(!is.na(Value))

  # Two-way ANOVA: Fraction x Genotype
  model <- aov(Value ~ Fraction * Genotype, data = dat)

  anova_tab <- as.data.frame(summary(model)[[1]])
  anova_tab$Effect <- rownames(anova_tab)
  rownames(anova_tab) <- NULL

  names(anova_tab)[names(anova_tab) == "Df"] <- "DF"
  names(anova_tab)[names(anova_tab) == "Sum Sq"] <- "Sum_Sq"
  names(anova_tab)[names(anova_tab) == "Mean Sq"] <- "Mean_Sq"
  names(anova_tab)[names(anova_tab) == "F value"] <- "F_value"
  names(anova_tab)[names(anova_tab) == "Pr(>F)"] <- "P_value"

  anova_tab <- anova_tab %>%
    mutate(Marker = marker, .before = 1) %>%
    select(Marker, Effect, DF, Sum_Sq, Mean_Sq, F_value, P_value)

  emm <- emmeans(model, ~ Fraction * Genotype)

  # Four planned comparisons:
  # Col-0 crude vs Col-0 Mito, bou crude vs bou Mito,
  # Col-0 crude vs bou crude, Col-0 Mito vs bou Mito
  contrast_list <- list(
    "Col-0 crude vs Col-0 Mito" = c(1, -1, 0, 0),
    "bou crude vs bou Mito"     = c(0, 0, 1, -1),
    "Col-0 crude vs bou crude"  = c(1, 0, -1, 0),
    "Col-0 Mito vs bou Mito"    = c(0, 1, 0, -1)
  )

  pairwise <- contrast(emm, method = contrast_list, adjust = "bonferroni")
  pairwise <- as.data.frame(pairwise)

  pairwise <- pairwise %>%
    mutate(Marker = marker, .before = 1) %>%
    rename(
      Comparison = contrast,
      Estimate = estimate,
      SE = SE,
      DF = df,
      t_ratio = t.ratio,
      P_value_Bonferroni = p.value
    ) %>%
    mutate(
      Significant_0.05 = ifelse(P_value_Bonferroni < 0.05, "Yes", "No"),
      Significance = case_when(
        P_value_Bonferroni < 0.0001 ~ "****",
        P_value_Bonferroni < 0.001  ~ "***",
        P_value_Bonferroni < 0.01   ~ "**",
        P_value_Bonferroni < 0.05   ~ "*",
        TRUE ~ "ns"
      )
    )

  descriptive <- dat %>%
    group_by(Fraction, Genotype) %>%
    summarise(
      n = n(),
      Mean = mean(Value, na.rm = TRUE),
      SD = sd(Value, na.rm = TRUE),
      SEM = SD / sqrt(n),
      .groups = "drop"
    ) %>%
    mutate(Marker = marker, .before = 1)

  list(anova = anova_tab, pairwise = pairwise, descriptive = descriptive)
}

results <- lapply(markers, analyse_marker)

anova_results       <- bind_rows(lapply(results, `[[`, "anova"))
pairwise_results    <- bind_rows(lapply(results, `[[`, "pairwise"))
descriptive_results <- bind_rows(lapply(results, `[[`, "descriptive"))

settings <- data.frame(
  Item = c(
    "Analysis", "Model", "Factors", "Markers",
    "Comparison 1", "Comparison 2", "Comparison 3", "Comparison 4",
    "Multiple-testing correction", "Alpha"
  ),
  Value = c(
    "Two-way ANOVA + planned pairwise comparisons",
    "Value ~ Fraction * Genotype",
    "Fraction (crude/Mito) × Genotype (Col-0/bou)",
    paste(markers, collapse = ", "),
    "Col-0 crude vs Col-0 Mito",
    "bou crude vs bou Mito",
    "Col-0 crude vs bou crude",
    "Col-0 Mito vs bou Mito",
    "Bonferroni",
    "0.05"
  )
)

write_xlsx(
  list(
    ANOVA = anova_results,
    Pairwise = pairwise_results,
    Descriptive = descriptive_results,
    Settings = settings
  ),
  "stat.xlsx"
)

cat("\nANOVA:\n")
print(anova_results)

cat("\n\nPAIRWISE COMPARISONS:\n")
print(pairwise_results)

cat("\n\nDESCRIPTIVE STATISTICS:\n")
print(descriptive_results)

cat("\n\nResults saved to: stat.xlsx\n")

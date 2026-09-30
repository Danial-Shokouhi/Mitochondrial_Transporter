# FAD uptake statistics: two-way ANOVA with planned comparisons,
# plus separate Welch t-tests for transport rate and initial content.
#
# Input: Flavin_uptake.xlsx
#   Default sheet -- Genotype | Flavin | Incubation_time | Replicate | Net_uptake
#   Sheet "Uptake_rate"      -- Genotype | Flavin | Replicate | Uptake_rate
#   Sheet "Initial_content"  -- Genotype | Flavin | Replicate | Initial_Flavin_content
#
# Part 1 output: stat.xlsx  (ANOVA, Pairwise, Descriptive, Settings)
# Part 2 output: welch.xlsx (Uptake_rate_Stats, Uptake_rate_Descriptive,
#                             Initial_content_Stats, Initial_content_Descriptive, Settings)

required <- c("readxl", "dplyr", "emmeans", "writexl")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(
    "Missing R packages: ", paste(missing, collapse = ", "),
    "\nInstall them with:\n",
    "install.packages(c(", paste(sprintf('"%s"', missing), collapse = ", "), "))"
  )
}

library(readxl)
library(dplyr)
library(emmeans)
library(writexl)


# ============================================================
# Part 1 -- Two-way ANOVA + Bonferroni planned comparisons
# Net_uptake ~ Genotype * Incubation_time
# ============================================================

INPUT_FILE  <- "Flavin_uptake.xlsx"
OUTPUT_FILE <- "stat.xlsx"

if (!file.exists(INPUT_FILE)) {
  stop("Input file not found: ", INPUT_FILE)
}

dat <- read_excel(INPUT_FILE)

required_cols <- c("Genotype", "Flavin", "Incubation_time", "Replicate", "Net_uptake")
missing_cols <- setdiff(required_cols, names(dat))
if (length(missing_cols) > 0) {
  stop("These required columns are missing from Flavin_uptake.xlsx: ",
       paste(missing_cols, collapse = ", "))
}

dat <- dat %>%
  select(all_of(required_cols)) %>%
  mutate(
    Genotype = trimws(as.character(Genotype)),
    Flavin = trimws(as.character(Flavin)),
    Incubation_time = as.numeric(Incubation_time),
    Replicate = as.numeric(Replicate),
    Net_uptake = as.numeric(Net_uptake)
  ) %>%
  filter(
    !is.na(Genotype), !is.na(Flavin), !is.na(Incubation_time),
    !is.na(Replicate), !is.na(Net_uptake)
  ) %>%
  filter(Flavin == "FAD")  

expected_genotypes <- c("Col-0", "bou")
expected_flavins   <- c("FAD")
expected_times     <- c(10, 30)

bad_genotypes <- setdiff(unique(dat$Genotype), expected_genotypes)
bad_flavins   <- setdiff(unique(dat$Flavin), expected_flavins)
bad_times     <- setdiff(unique(dat$Incubation_time), expected_times)

if (length(bad_genotypes) > 0) stop("Unexpected Genotype values: ", paste(bad_genotypes, collapse = ", "))
if (length(bad_flavins) > 0)   stop("Unexpected Flavin values: ", paste(bad_flavins, collapse = ", "))
if (length(bad_times) > 0)     stop("Unexpected Incubation_time values: ", paste(bad_times, collapse = ", "))

dat <- dat %>%
  mutate(
    Genotype = factor(Genotype, levels = c("Col-0", "bou")),
    Flavin = factor(Flavin, levels = c("FAD")),
    Incubation_time = factor(Incubation_time, levels = c(10, 30), labels = c("10", "30"))
  )

descriptive <- dat %>%
  group_by(Flavin, Genotype, Incubation_time) %>%
  summarise(n = n(), Mean = mean(Net_uptake), SD = sd(Net_uptake),
            SEM = SD / sqrt(n), .groups = "drop") %>%
  arrange(Flavin, Genotype, Incubation_time)

analyse_flavin <- function(df, flavin_name) {

  df <- droplevels(df)

  model <- aov(Net_uptake ~ Genotype * Incubation_time, data = df)

  aov_tab <- as.data.frame(summary(model)[[1]])
  aov_tab$Effect <- rownames(aov_tab)
  rownames(aov_tab) <- NULL
  names(aov_tab)[names(aov_tab) == "Df"] <- "DF"
  names(aov_tab)[names(aov_tab) == "Sum Sq"] <- "Sum_Sq"
  names(aov_tab)[names(aov_tab) == "Mean Sq"] <- "Mean_Sq"
  names(aov_tab)[names(aov_tab) == "F value"] <- "F_value"
  names(aov_tab)[names(aov_tab) == "Pr(>F)"] <- "P_value"

  aov_out <- aov_tab %>%
    mutate(Flavin = flavin_name, .before = 1) %>%
    select(Flavin, Effect, DF, Sum_Sq, Mean_Sq, F_value, P_value)

  # Genotype comparison at each incubation time (Col-0 - bou)
  emm_genotype <- emmeans(model, ~ Genotype | Incubation_time)
  gt_contrast <- contrast(emm_genotype, method = list("Col-0 - bou" = c(1, -1)), adjust = "none")
  gt_df <- as.data.frame(gt_contrast) %>%
    mutate(
      Comparison = paste0("Col-0  ", as.character(Incubation_time), " min vs bou ",
                          as.character(Incubation_time), " min"),
      Flavin = flavin_name, .before = 1
    ) %>%
    select(Flavin, Comparison, estimate, SE, df, t.ratio, p.value, Incubation_time)
  gt_df <- gt_df %>% select(-Incubation_time)

  # Time comparison within genotype (10 - 30)
  emm_time <- emmeans(model, ~ Incubation_time | Genotype)
  time_contrast <- contrast(emm_time, method = list("10 - 30" = c(1, -1)), adjust = "none")
  time_df <- as.data.frame(time_contrast) %>%
    mutate(
      Comparison = case_when(
        as.character(Genotype) == "Col-0" ~ "Col-0 10 min vs Col-0 30 min",
        as.character(Genotype) == "bou"   ~ "bou 10 min vs bou 30 min",
        TRUE ~ as.character(Genotype)
      ),
      Flavin = flavin_name, .before = 1
    ) %>%
    select(Flavin, Comparison, estimate, SE, df, t.ratio, p.value, Genotype)
  time_df <- time_df %>% select(-Genotype)

  pairwise <- bind_rows(gt_df, time_df) %>%
    mutate(
      p_bonferroni = p.adjust(p.value, method = "bonferroni"),
      Significant_0.05 = ifelse(p_bonferroni < 0.05, "Yes", "No"),
      Significance = case_when(
        p_bonferroni < 0.0001 ~ "****",
        p_bonferroni < 0.001  ~ "***",
        p_bonferroni < 0.01   ~ "**",
        p_bonferroni < 0.05   ~ "*",
        TRUE ~ "ns"
      )
    ) %>%
    rename(Estimate = estimate, SE = SE, DF = df, t_ratio = t.ratio,
           Raw_p = p.value, Bonferroni_p = p_bonferroni)

  list(anova = aov_out, pairwise = pairwise)
}

results <- list()
for (flavin in expected_flavins) {
  df_f <- dat %>% filter(Flavin == flavin)

  counts <- df_f %>% count(Genotype, Incubation_time)
  if (nrow(counts) != 4 || any(counts$n < 2)) {
    warning("Incomplete or low-replicate design for ", flavin, ". Check the input data.")
  }

  results[[flavin]] <- analyse_flavin(df_f, flavin)
}

anova_results <- bind_rows(results$FAD$anova)
pairwise_results <- bind_rows(results$FAD$pairwise)

settings <- data.frame(
  Item = c("Analysis", "Model", "Experimental factors", "FAD comparisons",
          "Multiple-testing correction", "Correction scope", "Alpha"),
  Value = c(
    "Two-way ANOVA with planned post-hoc comparisons",
    "Net_uptake ~ Genotype * Incubation_time",
    "Genotype (Col-0, bou) × Incubation_time (10, 30 min)",
    "Col-0 10 vs bou 10; Col-0 30 vs bou 30; Col-0 10 vs Col-0 30; bou 10 vs bou 30",
    "Bonferroni", "Four planned comparisons within FAD", "0.05"
  ),
  stringsAsFactors = FALSE
)

write_xlsx(
  list(ANOVA = anova_results, Pairwise = pairwise_results,
       Descriptive = descriptive, Settings = settings),
  path = OUTPUT_FILE
)

cat("\nSaved:", OUTPUT_FILE, "\n\n")
cat("ANOVA results:\n")
print(anova_results)
cat("\nBonferroni-adjusted planned comparisons:\n")
print(pairwise_results)


# ============================================================
# Part 2 -- Welch t-tests: transport rate and initial flavin content
# Col-0 vs bou
# ============================================================

WELCH_OUTPUT_FILE <- "welch.xlsx"

run_welch <- function(df, value_col, analysis_name) {

  df <- df %>%
    mutate(
      Genotype = trimws(as.character(Genotype)),
      Flavin = trimws(as.character(Flavin)),
      Replicate = as.numeric(Replicate),
      value = as.numeric(.data[[value_col]])
    ) %>%
    filter(Genotype %in% c("Col-0", "bou"), Flavin %in% c("FAD"), !is.na(value))

  descriptive <- df %>%
    group_by(Flavin, Genotype) %>%
    summarise(n = n(), Mean = mean(value), SD = sd(value),
              SEM = SD / sqrt(n), .groups = "drop") %>%
    arrange(Flavin, Genotype)

  results <- list()
  for (flavin in c("FAD")) {
    x <- df %>% filter(Genotype == "Col-0", Flavin == flavin) %>% pull(value)
    y <- df %>% filter(Genotype == "bou", Flavin == flavin) %>% pull(value)

    if (length(x) < 2 || length(y) < 2) {
      stop("Not enough non-missing replicates for ", flavin, " in ", analysis_name,
           ". Need at least 2 observations per genotype.")
    }

    tt <- t.test(x, y, alternative = "two.sided", var.equal = FALSE)

    results[[flavin]] <- data.frame(
      Analysis = analysis_name, Flavin = flavin,
      Comparison = paste0("Col-0 ", flavin, " vs bou ", flavin),
      Col0_n = length(x), bou_n = length(y),
      Col0_mean = mean(x), Col0_SD = sd(x),
      bou_mean = mean(y), bou_SD = sd(y),
      Mean_difference_Col0_minus_bou = mean(x) - mean(y),
      t = unname(tt$statistic), DF = unname(tt$parameter),
      P_value_two_tailed = tt$p.value,
      CI_low = tt$conf.int[1], CI_high = tt$conf.int[2],
      stringsAsFactors = FALSE
    )
  }

  test_results <- bind_rows(results) %>%
    mutate(Significance = case_when(
      P_value_two_tailed < 0.0001 ~ "****",
      P_value_two_tailed < 0.001  ~ "***",
      P_value_two_tailed < 0.01   ~ "**",
      P_value_two_tailed < 0.05   ~ "*",
      TRUE ~ "ns"
    ))

  settings <- data.frame(
    Item = c("Analysis", "Experimental design", "Test", "Alternative",
            "Variance assumption", "Comparisons", "Alpha"),
    Value = c(
      analysis_name, "Unpaired / independent groups", "Welch two-sample t-test",
      "Two-tailed", "Unequal variances allowed (var.equal = FALSE)",
      "Col-0 FAD vs bou FAD", "0.05"
    ),
    stringsAsFactors = FALSE
  )

  list(Results = test_results, Descriptive = descriptive, Settings = settings)
}

uptake <- read_excel(INPUT_FILE, sheet = "Uptake_rate")
initial <- read_excel(INPUT_FILE, sheet = "Initial_content")

uptake_required <- c("Genotype", "Flavin", "Replicate", "Uptake_rate")
initial_required <- c("Genotype", "Flavin", "Replicate", "Initial_Flavin_content")

miss_uptake <- setdiff(uptake_required, names(uptake))
miss_initial <- setdiff(initial_required, names(initial))
if (length(miss_uptake) > 0) stop("Uptake_rate is missing: ", paste(miss_uptake, collapse = ", "))
if (length(miss_initial) > 0) stop("Initial_content is missing: ", paste(miss_initial, collapse = ", "))

uptake_stats <- run_welch(uptake, "Uptake_rate", "Flavin uptake rate")
initial_stats <- run_welch(initial, "Initial_Flavin_content", "Initial flavin content")

write_xlsx(
  list(
    Uptake_rate_Stats = uptake_stats$Results,
    Uptake_rate_Descriptive = uptake_stats$Descriptive,
    Initial_content_Stats = initial_stats$Results,
    Initial_content_Descriptive = initial_stats$Descriptive,
    Settings = uptake_stats$Settings
  ),
  path = WELCH_OUTPUT_FILE
)

cat("\nSaved:", WELCH_OUTPUT_FILE, "\n\n")
cat("Uptake-rate Welch tests:\n")
print(uptake_stats$Results)
cat("\nInitial-content Welch tests:\n")
print(initial_stats$Results)

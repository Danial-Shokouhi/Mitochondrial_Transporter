library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(writexl)
library(ggnewscale)

# Load mitochondrial metabolite data
df <- read_excel("Metabolites.xlsx", sheet = "Mito")
names(df) <- trimws(names(df))

# Reshape to long format and assign genotype from sample name
df_long <- df %>%
  pivot_longer(cols = -Metabolite, names_to = "Sample", values_to = "Value") %>%
  mutate(
    Value = as.numeric(Value),
    Group = ifelse(str_starts(Sample, "B"), "bou", "Col-0")
  ) %>%
  filter(!is.na(Value))

# Per-metabolite statistics (t-test, falling back to Wilcoxon if variance is zero)
metabolites <- unique(df_long$Metabolite)

results <- lapply(metabolites, function(met) {
  sub <- df_long %>% filter(Metabolite == met)
  col0 <- sub %>% filter(Group == "Col-0") %>% pull(Value)
  bou  <- sub %>% filter(Group == "bou") %>% pull(Value)

  pval <- tryCatch({
    t.test(col0, bou)$p.value
  }, error = function(e) {
    wilcox.test(col0, bou, exact = FALSE)$p.value
  })

  data.frame(
    Metabolite = met,
    mean_Col0 = mean(col0, na.rm = TRUE),
    mean_bou = mean(bou, na.rm = TRUE),
    fold_change = mean(bou, na.rm = TRUE) / mean(col0, na.rm = TRUE),
    p_value = pval
  )
})

stats_table <- bind_rows(results)
stats_table$padj <- p.adjust(stats_table$p_value, method = "BH")

df_stats <- df_long %>%
  left_join(stats_table, by = "Metabolite")

# Bar plot with individual points, faceted per metabolite
make_plot <- function(data, metabolites, nrow, ncol, title, outfile) {

  plot_data <- data %>%
    filter(Metabolite %in% metabolites) %>%
    mutate(Group = factor(Group, levels = c("bou", "Col-0")))

  # Per-facet bracket position, scaled to that metabolite's max value
  p_dat <- plot_data %>%
    group_by(Metabolite) %>%
    summarise(y_max = max(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      y_bracket = y_max * 1.15,
      y_text    = y_max * 1.25,
      label     = "",
      x         = 1,
      xend      = 2
    )

  p <- ggplot(plot_data, aes(x = Group, y = Value)) +
    stat_summary(
      aes(fill = Group), fun = mean, geom = "col",
      width = 0.6, color = "black", linewidth = 0.6
    ) +
    scale_fill_manual(values = c("Col-0" = "#707071", "bou" = "#C03830")) +
    ggnewscale::new_scale_fill() +   # second fill scale, for the jittered points below
    stat_summary(
      fun.data = mean_sdl, fun.args = list(mult = 1),
      geom = "errorbar", width = 0.2, linewidth = 0.6
    ) +
    geom_jitter(
      aes(fill = Group), shape = 21, color = "black", stroke = 0.6,
      width = 0.12, size = 2.5, alpha = 0.9
    ) +
    scale_fill_manual(values = c("Col-0" = "#D0D3D6", "bou" = "#F47671")) +
    facet_wrap(~Metabolite, nrow = nrow, ncol = ncol, scales = "free_y") +
    scale_y_continuous(breaks = function(x) pretty(x, n = 3)) +
    labs(title = title, x = "Genotype", y = "nmol/mg mtProtein") +
    theme_classic() +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 14),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.title.y = element_text(size = 16),
      axis.title.x = element_text(size = 16),
      axis.text.x  = element_text(size = 18),
      axis.text.y  = element_text(size = 18),
      legend.position = "none"
    ) +
    geom_segment(
      data = p_dat, aes(x = x, xend = xend, y = y_bracket, yend = y_bracket),
      inherit.aes = FALSE, linewidth = 0.6
    ) +
    geom_segment(
      data = p_dat, aes(x = x, xend = x, y = y_bracket, yend = y_bracket * 0.97),
      inherit.aes = FALSE, linewidth = 0.6
    ) +
    geom_segment(
      data = p_dat, aes(x = xend, xend = xend, y = y_bracket, yend = y_bracket * 0.97),
      inherit.aes = FALSE, linewidth = 0.6
    ) +
    geom_text(
      data = p_dat, aes(x = 1.5, y = y_text, label = label),
      inherit.aes = FALSE, size = 3.5, fontface = "bold"
    )

  ggsave(outfile, p, dpi = 1000, width = 13, height = 13)
  invisible(p)
}

main_metabolites <- c(
  "Glycine", "Valine", "Leucine", "Isoleucine", "Proline", "Serine",
  "Lipoic acid", "pyruvic acid", "Succinic acid", "Oxoglutaric acid",
  "NAD+", "NADH", "Fumaric acid"
)

make_plot(
  data = df_stats,
  metabolites = main_metabolites,
  nrow = 6, ncol = 3,
  title = "Selected Metabolites",
  outfile = "Figure1_Selected.png"
)

make_plot(
  data = df_stats,
  metabolites = setdiff(unique(df_stats$Metabolite), main_metabolites),
  nrow = 5, ncol = 5,
  title = "",
  outfile = "Figure2_Rest.png"
)

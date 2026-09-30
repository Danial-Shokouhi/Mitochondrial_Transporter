library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tibble)
library(ggrepel)
library(writexl)

# Load whole-cell metabolite data
df <- read_excel("Metabolites.xlsx", sheet = "WC")
names(df) <- trimws(names(df))
names(df)[1] <- "Metabolite"

# Reshape to long format and assign genotype from sample name
df_long <- df %>%
  pivot_longer(
    cols = -Metabolite, names_to = "Sample", values_to = "Value",
    values_drop_na = TRUE
  ) %>%
  mutate(Group = ifelse(grepl("^B", Sample), "Mutant", "WT"))

# Per-metabolite mean/SD/n by group
summary_stats <- df_long %>%
  group_by(Metabolite, Group) %>%
  summarise(
    mean = mean(Value, na.rm = TRUE),
    sd = sd(Value, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = Group, values_from = c(mean, sd, n))

stats <- summary_stats %>%
  mutate(log2FC = log2(mean_Mutant / mean_WT))

# t-test per metabolite, skipped (NA) if either group has zero variance
pvals <- df_long %>%
  group_by(Metabolite) %>%
  summarise(
    p_value = tryCatch({
      if (sd(Value[Group == "Mutant"], na.rm = TRUE) == 0 |
          sd(Value[Group == "WT"], na.rm = TRUE) == 0) {
        NA
      } else {
        t.test(Value ~ Group)$p.value
      }
    }, error = function(e) NA),
    .groups = "drop"
  )

final_df <- stats %>%
  left_join(pvals, by = "Metabolite") %>%
  mutate(
    padj = p.adjust(p_value, method = "BH"),
    neglog10 = -log10(padj),
    sig = padj < 0.05 & abs(log2FC) > 1.0
  )

write_xlsx(final_df, "Metabolite_Statistics.xlsx")

# Volcano plot
ggplot(final_df, aes(x = log2FC, y = neglog10)) +
  geom_point(
    aes(fill = sig), size = 6, shape = 21, color = "black", stroke = 1.3
  ) +
  geom_text_repel(
    data = subset(final_df, sig == TRUE),
    aes(label = Metabolite),
    size = 4, fontface = "bold", max.overlaps = 50,
    box.padding = 1.1, point.padding = 0.7,
    segment.color = "black", segment.size = 0.6, segment.alpha = 0.4,
    min.segment.length = 0
  ) +
  geom_vline(xintercept = 0, linewidth = 1.3) +
  geom_hline(
    yintercept = -log10(0.05), linetype = "dashed",
    linewidth = 0.8, color = "black", alpha = 0.8
  ) +
  geom_text(
    data = data.frame(y = pretty(final_df$neglog10), x = 0),
    aes(x = x, y = y, label = round(y, 1)),
    hjust = -0.5, size = 4, fontface = "bold"
  ) +
  scale_fill_manual(values = c("grey80", "#C03830")) +
  coord_cartesian(
    xlim = c(
      -max(abs(final_df$log2FC), na.rm = TRUE),
      max(abs(final_df$log2FC), na.rm = TRUE)
    )
  ) +
  theme_classic(base_size = 36) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    axis.text.x = element_text(size = 18, face = "bold"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.position = "none"
  ) +
  annotate(
    "text", x = max(final_df$log2FC) * -0.92, y = -log10(0.05),
    label = "P = 0.05", vjust = -0.6, size = 5, fontface = "bold"
  ) +
  labs(
    x = expression(bold(Log[2] * FC)),
    y = expression(bold(-log[10](Padj))),
    title = "Whole cell Metabolites"
  )

ggsave("Volcano_WC.png", dpi = 1000, width = 10, height = 6.9, units = "in")

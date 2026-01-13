library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)

df <- read_csv("/Users/cartersale/Documents/Pose_Dynamics/projects/mirror_game/data/rqa/mirror_game_crqa_results.csv")

# Add trial_order and make grouping vars factors
df <- df %>%
  mutate(
    trial_order = as.numeric(Trial),
    Pair       = factor(Pair),
    Condition  = factor(Condition),
    keypoint   = factor(keypoint)
  )

metrics <- c(
  "perc_recur",
  "perc_determ",
  "entropy",
  "complexity",
  "maxl_found",
  "mean_line_length",
  "std_line_length",
  "laminarity",
  "trapping_time",
  "vmax",
  "divergence"
)

results <- purrr::map_df(metrics, function(metric) {
  
  form <- as.formula(
    paste0(metric, " ~ Condition + trial_order + (1 | Pair/Trial) + (1 | keypoint)")
  )
  
  model <- tryCatch(
    lmer(form, data = df, REML = TRUE),
    error = function(e) {
      message("Error fitting metric: ", metric)
      message(e$message)
      return(NULL)
    }
  )
  
  if (is.null(model)) {
    return(tibble(
      metric   = metric,
      term     = NA_character_,
      estimate = NA_real_,
      std.error = NA_real_,
      df       = NA_real_,
      p.value  = NA_real_
    ))
  }
  
  broom.mixed::tidy(model, effects = "fixed") %>%
    mutate(metric = metric) %>%
    select(metric, term, estimate, std.error, df, p.value)
})

results
write_csv(results, "/Users/cartersale/Documents/Pose_Dynamics/projects/mirror_game/data/rqa/mixed_model_results_crqa.csv")



results_plot <- results %>%
  filter(term %in% c("Conditionf2f", "Conditionuni")) %>%
  mutate(
    condition = recode(term,
                       "Conditionf2f" = "f2f",
                       "Conditionuni" = "uni"),
    metric = factor(metric, levels = metrics)
  )

results_plot <- results_plot %>%
  mutate(
    lower = estimate - 1.96 * std.error,
    upper = estimate + 1.96 * std.error
  )



library(ggplot2)

p <- ggplot(results_plot,
            aes(x = metric, y = estimate,
                ymin = lower, ymax = upper,
                color = condition)) +
  geom_pointrange(position = position_dodge(width = 0.6),
                  size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Condition Effects on RQA Metrics",
    x = "RQA Metric",
    y = "Mixed Model Estimate (relative to b2b)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p

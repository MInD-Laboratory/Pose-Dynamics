library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)

# -----------------------------
# Load + basic prep
# -----------------------------
df <- read_csv("/Users/cartersale/Research/Projects/Pose_Dynamics/projects/mirror_game/data/rqa/mirror_game_crqa_results.csv")

df <- df %>%
  mutate(
    Pair      = factor(Pair),
    Trial     = factor(Trial),
    Condition = factor(Condition),
    keypoint  = factor(keypoint),
    # define PairTrial explicitly once
    PairTrial = interaction(Pair, Trial, drop = TRUE)
  )

# If Trial is already the ordinal trial number within each Pair, you can do:
# Otherwise, this creates an order within Pair based on Trial levels.
df <- df %>%
  group_by(Pair) %>%
  mutate(trial_order = as.numeric(factor(Trial, levels = unique(Trial)))) %>%
  ungroup()

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

# -----------------------------
# Helper: choose random effects safely
# -----------------------------
choose_formula <- function(metric, data) {
  n_obs <- nrow(data)
  n_levels_pairtrial <- nlevels(data$PairTrial)

  use_pairtrial <- n_levels_pairtrial < n_obs

  form <- if (use_pairtrial) {
    as.formula(paste0(metric, " ~ Condition + trial_order + (1 | Pair) + (1 | PairTrial)"))
  } else {
    as.formula(paste0(metric, " ~ Condition + trial_order + (1 | Pair)"))
  }

  list(form = form, re_structure = ifelse(use_pairtrial, "Pair + Pair:Trial", "Pair only"))
}

fit_one <- function(data, metric, scope, keypoint_label) {
  cf <- choose_formula(metric, data)

  model <- tryCatch(
    lmer(cf$form, data = data, REML = TRUE),
    error = function(e) {
      message("Error fitting ", toupper(scope), " metric: ", metric, " | keypoint: ", keypoint_label)
      message(e$message)
      return(NULL)
    }
  )

  if (is.null(model)) {
    return(tibble(
      scope = scope,
      keypoint = keypoint_label,
      metric = metric,
      re_structure = cf$re_structure,
      term = NA_character_,
      estimate = NA_real_,
      std.error = NA_real_,
      df = NA_real_,
      p.value = NA_real_
    ))
  }

  broom.mixed::tidy(model, effects = "fixed") %>%
    mutate(
      scope = scope,
      keypoint = keypoint_label,
      metric = metric,
      re_structure = cf$re_structure
    ) %>%
    select(scope, keypoint, metric, re_structure, term, estimate, std.error, df, p.value)
}

# -----------------------------
# 1) SUBSET (average across keypoints) models
# -----------------------------
df_subset <- df %>%
  group_by(Pair, Trial, PairTrial, Condition, trial_order) %>%
  summarise(across(all_of(metrics), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(
    Pair = factor(Pair),
    Trial = factor(Trial),
    PairTrial = factor(PairTrial),
    Condition = factor(Condition)
  )

subset_results <- purrr::map_df(metrics, function(metric) {
  fit_one(df_subset, metric, scope = "subset", keypoint_label = "subset")
})

write_csv(
  subset_results,
  "/Users/cartersale/Research/Projects/Pose_Dynamics/projects/mirror_game/data/rqa/mixed_model_results_crqa_subset.csv"
)

# -----------------------------
# 2) PER-KEYPOINT models
# -----------------------------
per_keypoint_results <- purrr::map_df(metrics, function(metric) {
  purrr::map_df(levels(df$keypoint), function(kp) {
    df_kp <- df %>%
      filter(keypoint == kp) %>%
      mutate(
        Pair = factor(Pair),
        Trial = factor(Trial),
        PairTrial = factor(PairTrial),
        Condition = factor(Condition)
      )

    fit_one(df_kp, metric, scope = "per_keypoint", keypoint_label = as.character(kp))
  })
})

write_csv(
  per_keypoint_results,
  "/Users/cartersale/Research/Projects/Pose_Dynamics/projects/mirror_game/data/rqa/mixed_model_results_crqa_per_keypoint.csv"
)

# -----------------------------
# 3) Combined output
# -----------------------------
all_results <- bind_rows(subset_results, per_keypoint_results)

write_csv(
  all_results,
  "/Users/cartersale/Research/Projects/Pose_Dynamics/projects/mirror_game/data/rqa/mixed_model_results_crqa_ALL.csv"
)

# -----------------------------
# 4) Plot (subset by default)
# -----------------------------
results_plot <- all_results %>%
  filter(scope == "subset") %>%
  filter(term %in% c("Conditionf2f", "Conditionuni")) %>%
  mutate(
    condition = recode(term,
                       "Conditionf2f" = "f2f",
                       "Conditionuni" = "uni"),
    metric = factor(metric, levels = metrics),
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
    title = "Condition Effects on CRQA Metrics (Subset-Averaged)",
    x = "CRQA Metric",
    y = "Mixed model estimate (relative to back-to-back)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p
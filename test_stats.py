import pandas as pd
from projects.MATB import stats_utils

lin_df = pd.read_csv("projects/MATB/data/processed/linear_metrics/procrustes_global_linear.csv")
res = stats_utils.run_rpy2_lmer(lin_df, "pupil_metric_vel_rms")
print(res[0])

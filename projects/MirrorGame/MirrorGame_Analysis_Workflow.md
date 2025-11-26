# Comprehensive Analysis Workflow: Mirror Game Pose Dynamics

Complete preprocessing and analysis pipeline for the Mirror Game notebook.

---

## 1. SETUP & DATA LOADING

### 1.1 Library Imports
**Functions/Utilities:**
- Standard libraries: `sys`, `os`, `pathlib`, `re`, `collections.defaultdict`
- Numerical: `numpy`, `pandas`, `sklearn.decomposition.PCA`
- Custom modules from `pose_dynamics`:
  - `preprocessing.*`
  - `geometry_utils.*`
  - `io.*`
  - `normalization.*`
  - `rqa.crossRQA.crossRQA`
  - `rqa.multivariateRQA.multivariateRQA`

**Purpose:** Set up environment and import analysis tools

### 1.2 Configuration Parameters
**Parameters:**
- `ALLOW_ROTATION = True` - Enable rotation in Procrustes alignment
- `ALLOW_SCALE = False` - Disable scaling in Procrustes alignment
- `REF_IDX = None` - Center all keypoints (no single reference point)
- `N_COMPONENTS = 14` - Number of principal components to extract
- `TARGET_RATE = 30.0` - Target sampling rate (Hz)

**Purpose:** Define global analysis parameters

### 1.3 File Discovery and Grouping
**Operation:** Regex parsing of filenames

**Pattern:** `P\d{3}_T\d+_(P1|P2)_pose_3d.csv`

**Functions:**
- `re.compile()` - Pattern matching
- `defaultdict(dict)` - Organize files by (pair, trial)

**Under the hood:**
- Extracts: Pair ID (P001-P999), Trial number (T1-T12), Party (P1/P2)
- Creates mapping: `{(pair, trial): {'P1': filepath, 'P2': filepath}}`

**Purpose:** Match dyad partners for paired analysis

---

## 2. TEMPORAL ALIGNMENT

### 2.1 Data Loading per Trial
**Functions:** `pd.read_csv()`

**Columns expected:** `x0, y0, z0, ..., xN, yN, zN` (3D keypoint coordinates)

**Purpose:** Load raw 3D pose data

### 2.2 Temporal Resampling
**Function:** `normalization.resample()`

**Parameters:**
- `target_rate = 30` Hz
- `dt_col = "dt_ms"` - Time delta column

**Under the hood:**
```python
# From normalization.py:
1. Cumulative sum of dt_ms → time_s
2. Set first row to t=0
3. Create uniform timeline: np.arange(start, end, 1/target_rate)
4. Interpolate to uniform grid with df.interpolate("linear")
```

**Purpose:** Ensure uniform 30 Hz sampling across all recordings

### 2.3 Temporal Alignment of Dyads
**Function:** `normalization.align_pair()`

**Under the hood:**
```python
1. Find common time overlap:
   start = max(df1["time_s"].min(), df2["time_s"].min())
   end = min(df1["time_s"].max(), df2["time_s"].max())
2. Restrict both to [start, end]
3. Truncate to minimum length: min(len(df1_aligned), len(df2_aligned))
```

**Purpose:** Ensure P1 and P2 have exactly matching time series length

### 2.4 Column Ordering
**Function:** `preprocessing.order_xyz_triplets()`

**Under the hood:**
```python
1. Extract numeric indices from column names (x0, y0, z0, ...)
2. Sort indices
3. Return ordered list: [x0, y0, z0, x1, y1, z1, ...]
```

**Purpose:** Standardize column order for downstream processing

---

## 3. SPATIAL ALIGNMENT (PROCRUSTES)

### 3.1 Stage 1: Per-Trial Centering
**Function:** `preprocessing.align_keypoints_3d()`

**Parameters:**
- `ref_idx = None` - Center on mean
- `use_procrustes = False` - Centering only (first pass)

**Under the hood:**
```python
1. Reshape to (T, n_points, 3)
2. If ref_idx is None:
   coords -= coords.mean(axis=1, keepdims=True)  # Center each frame
3. Return flattened (T, n_points*3)
```

**Purpose:** Remove translation variability within each trial

### 3.2 Per-Trial Canonicalization
**Function:** `preprocessing.canonicalise_mean_pose()`

**Under the hood:**
```python
1. Center on pelvis: P -= P[0]
2. Define body axes:
   - x_body = P[11] - P[10]  # Left→Right shoulder (X-axis)
   - y_body = P[3] - P[0]    # Pelvis→Neck (Z-axis, up)
3. Orthogonalize:
   - x = x_body / ||x_body||
   - y = y_body - (x·y_body)x; y /= ||y||
   - z = x × y  (forward direction)
4. Ensure right-handed: if (x×y)·z < 0, flip z
5. Ensure forward-facing: if z[1] < 0, flip z and recompute y
6. Rotation matrix: R = [x, y, z]
7. Return: P @ R
```

**Purpose:** Align each trial's mean pose to a canonical body-centered coordinate system

### 3.3 Global Template Construction
**Function:** `preprocessing.build_template_with_canonicalisation()`

**Under the hood:**
```python
1. For each trial:
   - Compute mean pose over time
   - Canonicalize mean pose
2. Initial template = mean of all canonicalized means
3. Refinement loop:
   - For each canonicalized mean:
     - Compute Procrustes transform to template
     - Apply transform: (mean @ R) * s + t
   - Update template = mean of aligned poses
```

**Purpose:** Create a reference skeleton representing average body configuration across all trials

### 3.4 Stage 2: Frame-by-Frame Alignment to Global Template
**Function:** `preprocessing.canonicalise_trial()`

**Parameters:**
- `global_template` - The reference skeleton from 3.3
- `allow_rotation = True`
- `allow_scale = False`

**Under the hood:**
```python
1. Compute trial mean pose
2. Find rigid transform aligning trial_mean to global_template:
   - compute_procrustes_transform_3d():
     a. Center both: Tc = template - mean(template)
                     Mc = trial_mean - mean(trial_mean)
     b. Normalize: Tc_n = Tc / ||Tc||, Mc_n = Mc / ||Mc||
     c. SVD: H = Mc_n.T @ Tc_n
             U, _, Vt = svd(H)
             R = U @ Vt
     d. Handle reflection: if det(R) < 0, flip Vt[-1]
     e. Scale: s = 1.0 (disabled)
     f. Translation: t = mean(template) - s * mean(trial_mean) @ R
3. Apply SAME transform to ALL frames in trial:
   aligned = (seq @ R) + t
```

**Purpose:** Orient every frame to face the same direction as the global template, removing orientation variability

---

## 4. DIMENSIONALITY REDUCTION (PCA)

### 4.1 Data Stacking
**Operation:** `np.vstack(all_aligned)`

**Under the hood:**
- Stack all frames from all trials: (total_frames, n_points*3)
- For Mirror Game: n_points = 38 keypoints → 114 dimensions

**Purpose:** Create single data matrix for PCA

### 4.2 Mean Centering
**Operation:**
```python
X_mean = X.mean(0, keepdims=True)
X_mc = X - X_mean
```

**Purpose:** Center data for PCA (required for variance interpretation)

### 4.3 PCA Computation
**Function:** `sklearn.decomposition.PCA(n_components=14)`

**Under the hood:**
```python
pca.fit(X_mc)
# Computes:
# - eigenvectors (principal components)
# - eigenvalues (variance explained)
# - mean (for inverse transform)
```

**Purpose:** Extract 14 principal movement patterns

### 4.4 Projection to PC Space
**Function:** `pca.transform(X_mc)`

**Output:** PC scores for each frame (total_frames, 14)

**Purpose:** Represent high-dimensional pose as low-dimensional trajectory

### 4.5 Data Organization
**Operations:**
- Create mapping: trial_index → (pair_trial, party)
- Split PC scores back into per-trial arrays
- Save to CSV: `pc_scores[trial_idx].csv` with columns PC1-PC14

**Purpose:** Maintain trial structure for paired analyses

---

## 5. CROSS-RECURRENCE QUANTIFICATION ANALYSIS (CRQA)

### 5.1 Parameter Configuration
**Parameters:**
```python
RQA_PARAMS = {
    'eDim': 4,           # Embedding dimension
    'tLag': 10,          # Time lag (frames)
    'radius': 0.20,      # Recurrence threshold (20%)
    'minl': 2,           # Minimum diagonal line length
    'rescaleNorm': 1,    # Mean-distance rescaling
    'norm': 2            # Z-score normalization
}
```

**Purpose:** Define time-delay embedding and recurrence parameters

### 5.2 Per-PC CRQA Analysis
**Function:** `pc_rqa_utils.run_pc_rqa_analysis()`

**For each dyad and each PC (1-14):**

#### 5.2.1 Normalize Time Series
**Function:** `norm_utils.normalize_data(x, norm=2)`

**Under the hood:**
```python
# Z-score normalization:
x_normalized = (x - mean(x)) / std(x)
y_normalized = (y - mean(y)) / std(y)
```

#### 5.2.2 Time-Delay Embedding
**Function:** `rqa_utils_cpp.rqa_dist()`

**Parameters:** `dim=4, lag=10`

**Under the hood (C++ implementation):**
```cpp
// For each time series:
// Construct embedded vectors:
// v[t] = [x[t], x[t+lag], x[t+2*lag], x[t+3*lag]]
//
// For cross-RQA, compute distance matrix:
// D[i,j] = ||v_x[i] - v_y[j]||  (Euclidean distance)
```

**Purpose:** Capture temporal dependencies and compute cross-distance matrix

#### 5.2.3 Recurrence Matrix Construction
**Function:** `rqa_utils_cpp.rqa_stats()`

**Parameters:** `rescale=1, rad=0.20, diag_ignore=0, rqa_mode="cross"`

**Under the hood (C++ implementation):**
```cpp
1. Rescale distances:
   if rescale == 1:
       threshold = radius * mean(D)

2. Binarize:
   RP[i,j] = 1 if D[i,j] < threshold, else 0

3. Diagonal exclusion:
   // For cross-RQA, diag_ignore=0 (keep all)
```

#### 5.2.4 RQA Metrics Extraction
**Computed metrics:**

1. **Recurrence Rate (%REC):** Percentage of recurrent points
   - Formula: `sum(RP) / size(RP) * 100`

2. **Determinism (%DET):** Percentage of recurrent points forming diagonal lines
   - Finds diagonal lines ≥ minl
   - Formula: `sum(diagonal_points) / sum(RP) * 100`

3. **Mean Line Length (L):** Average length of diagonal lines
   - Formula: `sum(line_lengths) / count(lines)`

4. **Max Line Length (L_max):** Longest diagonal line found

5. **Entropy (ENTR):** Shannon entropy of line length distribution
   - Formula: `-sum(p_i * log(p_i))`

6. **Laminarity (LAM):** Percentage forming vertical lines
   - Measures temporal trapping

7. **Trapping Time (TT):** Average length of vertical lines

8. **Divergence (DIV):** `1 / L_max`
   - Measures system divergence

9. **Trend:** Regression slope of recurrence density along diagonals

**Purpose:** Quantify synchronization patterns between P1 and P2

### 5.3 Additional Linear Metrics
**Computed for each participant:**

1. **Standard Deviation (SD):** `np.std(x)`
2. **Mean Velocity:** `np.mean(np.abs(np.diff(x)))`
3. **Range:** `np.ptp(x)` (peak-to-peak)

**Purpose:** Capture linear movement characteristics

### 5.4 Condition Merging
**Function:** `merge_pc_rqa_with_conditions()`

**Under the hood:**
```python
1. Parse pair_trial: "P001_T3" → Pair=1, Trial=3
2. Load conditions CSV with block structure
3. Melt wide to long format (12 trials × 3 conditions)
4. Assign Leader/Follower roles:
   - Block 1: uses block1_lead
   - Block 2: opposite of block1_lead
5. Merge with CRQA results on (Pair, Trial)
```

**Purpose:** Add experimental condition (Low/Medium/High) and role information

---

## 6. MULTIVARIATE RECURRENCE ANALYSIS (MdRQA)

### 6.1 Whitening Transformation
**Function:** Custom `whiten()`

**Under the hood:**
```python
1. Center: Xc = X - mean(X, axis=0)
2. Covariance: C = cov(Xc, rowvar=False)
3. Whitening matrix: W = inv(sqrtm(C))
4. Apply: X_white = Xc @ W
```

**Purpose:** Decorrelate PC dimensions and normalize variance

### 6.2 MdRQA Configuration
**Parameters:**
```python
K_MDRQA = 8  # Number of PCs to include
norm = 'zscore'  # Normalization method
```

### 6.3 Auto-MdRQA (Individual)
**Function:** `multivariateRQA(X, params, mode='auto')`

**Input:** First K=8 PCs for each participant

**Under the hood:**
```python
1. Normalize: X_norm = normalize_data(X, 'zscore')
2. Whiten: X_white = whiten(X_norm)
3. Multivariate distance:
   rqa_dist_multivariate(X_white, X_white)
   # Computes: D[i,j] = ||X[i,:] - X[j,:]||  (across all K dims)
4. RQA stats with Theiler window (tw=1) to exclude main diagonal
```

**Purpose:** Quantify auto-recurrence in individual full-body coordination

### 6.4 Cross-MdRQA (Dyadic)
**Function:** `multivariateRQA([X, Y], params, mode='cross')`

**Input:** K=8 PCs from P1 and P2 (matched length)

**Under the hood:**
```python
1. Normalize both independently
2. Whiten both
3. Cross-distance:
   D[i,j] = ||X_P1[i,:] - X_P2[j,:]||
4. RQA stats with diag_ignore=0 (cross-recurrence)
```

**Purpose:** Quantify interpersonal coordination in K-dimensional PC space

### 6.5 Joint-Space MdRQA
**Function:** `multivariateRQA(Z, params, mode='auto')`

**Input:** Concatenated `Z = [P1 PCs | P2 PCs]` → 2K dimensions

**Under the hood:**
```python
1. Concatenate: Z = np.concatenate([X_P1, X_P2], axis=1)
   # Shape: (T, 16) for K=8
2. Whiten joint space
3. Auto-RQA on concatenated state vector
```

**Purpose:** Analyze recurrence of joint dyadic configurations

---

## 7. STATISTICAL ANALYSIS

### 7.1 Linear Metrics Analysis

#### 7.1.1 Data Preparation
**Function:** `stats_utils.prepare_leader_follower_data()`

**Under the hood:**
```python
1. Extract Leader/Follower columns from conditions
2. Create separate SD and velocity columns for each role
3. Convert to long format:
   - Each row: (Pair, Trial, Condition, Role, Value)
```

#### 7.1.2 Aggregation Across PCs
**Function:** `stats_utils.aggregate_across_pcs()`

**Under the hood:**
```python
groupby_cols = ['Pair', 'Trial', 'Condition', 'Role']
df_agg = df.groupby(groupby_cols).mean()
# Averages SD and velocity across all 14 PCs
```

#### 7.1.3 Two-Way Repeated Measures ANOVA
**Function:** `stats_utils.run_rm_anova_2way()`

**Design:** 3 (Condition: L/M/H) × 2 (Role: Leader/Follower)

**Under the hood (using rpy2 or statsmodels):**
```python
# If rpy2 available:
formula = 'Value ~ Condition * Role + Error(Pair/(Condition*Role))'
aov_result = aov(formula, data)

# Fallback to statsmodels:
aovrm = AnovaRM(df, dv='Value', subject='Pair',
                within=['Condition', 'Role'])
```

**Tests:**
1. Main effect of Condition
2. Main effect of Role
3. Condition × Role interaction

**Effect sizes:** Partial eta-squared

#### 7.1.4 Post-hoc Tests
**Function:** `stats_utils.bonferroni_pairwise()`

**For significant interactions:**
```python
# Simple effects by Role:
- Leaders: L vs M, L vs H, M vs H
- Followers: L vs M, L vs H, M vs H

# Bonferroni correction: alpha / n_comparisons
```

**Output:** Pairwise t-tests with corrected p-values

#### 7.1.5 Per-Component Analysis
**Function:** `stats_utils.run_pc_specific_anova()`

**For each PC (1-14) separately:**
```python
1. Subset data to PC_i
2. Run 2-way RM-ANOVA (Condition × Role)
3. Store F, p, effect size
4. Identify significant PCs
```

**Purpose:** Determine which movement patterns show condition/role effects

### 7.2 CRQA Metrics Analysis

#### 7.2.1 Metric Detection
**Columns analyzed:**
- `perc_recur` (Recurrence Rate)
- `perc_determ` (Determinism)
- `mean_line_length`
- `entropy`
- `laminarity`
- Plus others from RQA output

#### 7.2.2 Aggregate Analysis
**Aggregation:** Mean across all PCs within (Pair, Trial, Condition)

**Test:** One-way RM-ANOVA
```python
formula = 'Metric ~ Condition + Error(Pair/Condition)'
```

**Correction:** Bonferroni across multiple metrics

#### 7.2.3 PC-Specific Analysis
**For each metric and each PC:**
```python
1. Run 1-way RM-ANOVA (Condition)
2. Identify significant PC×Metric combinations
3. Post-hoc with Holm-Bonferroni correction
```

### 7.3 MdRQA Analysis
**Similar structure to CRQA:**
1. Auto-MdRQA metrics (per participant)
2. Cross-MdRQA metrics (dyadic)
3. Joint-space metrics
4. One-way RM-ANOVA for Condition effects
5. Post-hoc comparisons

---

## 8. VISUALIZATION

### 8.1 Keypoint Visualization
**Function:** `keypoint_viz` functions

**Operations:**
- Plot skeleton structure with connections
- Overlay P1 and P2
- Show temporal evolution

### 8.2 Principal Movement Visualization
**Custom plotting functions:**

**For each PC:**
```python
1. Find min/max PC scores across all data
2. Amplify to target RMS displacement
3. Reconstruct poses:
   P_extreme = X_mean + extreme_score * pca.components_[pc]
4. Reshape to (n_points, 3)
5. Project to 2D views (front, side)
6. Draw skeleton with edges
7. Overlay min/max in different colors
```

**Output:** 15-panel figure showing movement patterns for PC1-PC15

### 8.3 Leader/Follower Comparisons
**Plotting:**
- Line plots with SEM shading
- Condition-wise comparisons
- Separate panels for each PC
- Bar plots for aggregated metrics

### 8.4 Statistical Results Visualization
**Function:** `stats_utils.barplot_ax()`

**Under the hood:**
```python
1. Bar plot: conditions on x-axis, metric on y-axis
2. Error bars: Standard Error of Mean
3. Significance brackets:
   - Compute bracket heights avoiding overlap
   - Add stars: * p<.05, ** p<.01, *** p<.001
4. Style: Bold labels, clean spines
```

---

## SUMMARY OF COMPLETE WORKFLOW

**Raw Data → Final Analysis:**

1. **Temporal Alignment** (30 Hz resampling, dyad synchronization)
2. **Spatial Alignment** (Centering → Canonicalization → Global Procrustes)
3. **Dimensionality Reduction** (PCA to 14 components)
4. **Recurrence Analysis:**
   - Univariate CRQA (per PC, 14 analyses per dyad)
   - Multivariate MdRQA (8-PC subspace, 3 modes: auto/cross/joint)
5. **Statistical Testing:**
   - Linear metrics (SD, velocity)
   - RQA metrics (9 recurrence measures)
   - Condition effects (Low/Medium/High tempo)
   - Role effects (Leader/Follower)
6. **Visualization** (Movement patterns, statistical results)

---

## KEY DATA TRANSFORMATIONS

**Dimensional Evolution:**
- **Raw:** 3D coordinates (38 keypoints × 3 = 114D)
- **Aligned:** Centered & canonicalized (114D, orientation-normalized)
- **PC Space:** 14D principal components
- **Embedded:** 4 lags × 14 PCs = 56D (for time-delay embedding per PC)
- **Recurrence:** Binary matrices (N×N per analysis)
- **Final:** 13 scalar RQA metrics per analysis

**Analysis Count per Trial:**
- 14 per-PC CRQA analyses (P1 vs P2)
- 2 auto-MdRQA (P1, P2 individually)
- 1 cross-MdRQA (P1 vs P2)
- 1 joint-space MdRQA (concatenated state)
- **Total:** 18 recurrence analyses per trial

**Statistical Tests:**
- Aggregate metrics: 2-way RM-ANOVA (Condition × Role)
- Per-PC metrics: 14 separate 2-way RM-ANOVAs
- MdRQA metrics: 1-way RM-ANOVA (Condition)
- Post-hoc: Bonferroni-corrected pairwise comparisons

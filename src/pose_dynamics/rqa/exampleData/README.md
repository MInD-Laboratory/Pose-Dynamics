# Example Data for RQA Analysis

This directory contains example datasets for testing Recurrence Quantification Analysis (RQA) methods, including traditional univariate RQA, cross-RQA, multivariate RQA, and diagonal recurrence profiles (DRP).

## Time Series Files:

### Physiological/Experimental Data:

1. **PostureData.csv** - Human postural sway data (1800 points)
   - 2 columns: Center of pressure coordinates (x, y) in millimeters
   - Recorded during quiet standing balance task
   - Suitable for: Traditional RQA, Cross-RQA, Multivariate RQA, DRP
   - Example use: Balance control analysis, postural stability assessment

2. **RockingChairData.csv** - Synchronized rocking chair movements (3598 points)
   - 2 columns: Position data from two synchronized rocking chairs
   - Demonstrates interpersonal coordination dynamics
   - Suitable for: Cross-RQA analysis of coordination patterns including cross-DRP
   - Example use: Social motor coordination, synchronization analysis

### Categorical/Discrete Data:

3. **Elvis.csv** - Categorical time series data (120 points)
   - 2 columns: Categorical state values (tab-separated)
   - Represents discrete state transitions over time
   - Suitable for: Categorical RQA with minimal embedding
   - Example use: State transition analysis, categorical recurrence patterns

### Synthetic Data:

4. **WhiteNoiseData.csv** - White noise time series (3599 points)
   - 2 columns: Independent Gaussian white noise signals
   - Random, uncorrelated data for comparison/control
   - Suitable for: Baseline comparison, noise analysis
   - Example use: Control condition, testing RQA sensitivity to structure

### Chaotic Systems Data:

5. **lorenz_chaotic_xyz.csv** - Standard chaotic Lorenz system (1500 points)
   - 4 columns: time, x, y, z coordinates (σ=10, ρ=28, β=8/3)
   - Classic chaotic attractor with known dynamics
   - Initial condition: [1.0, 1.0, 1.0]
   - Suitable for: Multivariate RQA (ideal test case)
   - Example use: Chaotic dynamics analysis, multivariate method validation

6. **lorenz_chaotic_xy.csv** - Same Lorenz system, 2D subset (1500 points)
   - 3 columns: time, x, y coordinates only
   - For testing 2D multivariate RQA
   - Suitable for: 2D multivariate RQA, traditional RQA comparison

7. **lorenz_chaotic_xyz_2.csv** - Second Lorenz trajectory (1500 points)
   - Same parameters as lorenz_chaotic_xyz.csv
   - Different initial condition: [1.1, 1.1, 1.1]
   - Suitable for: Multivariate cross-RQA between similar systems
   - Example use: Comparing trajectories of same system with different ICs

8. **lorenz_hyperchaotic_xyz.csv** - Hyperchaotic Lorenz (1000 points)
   - 4 columns: time, x, y, z coordinates (σ=10, ρ=35, β=8/3)
   - Higher ρ parameter creates more complex dynamics
   - Suitable for: Advanced multivariate RQA, complex dynamics analysis
   - Example use: Hyperchaotic system analysis, parameter sensitivity studies

## Usage Examples:

### Traditional RQA (Single Variable):
```python
import pandas as pd
from autoRQA import autoRQA

# Example 1: Postural sway analysis
data = pd.read_csv('exampleData/PostureData.csv', header=None)
x_position = data.iloc[:, 0].values  # Use x-coordinate

params = {
    'norm': 1,                          # Normalize to unit interval
    'eDim': 3,                          # Embedding dimension
    'tLag': 15,                         # Time lag
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.1,                      # Recurrence radius
    'tw': 2,                            # Theiler window
    'minl': 2,                          # Minimum line length
    'plotMode': 'rp-timeseries',        # Plot type
    'pointSize': 2,
    'saveFig': False,
    'showMetrics': True,
    'doStatsFile': False
}

td, rs, mats, err_code = autoRQA(x_position, params)

# Example 2: Categorical data analysis
elvis_data = pd.read_csv('exampleData/Elvis.csv', header=None, sep='\t')
categories = elvis_data.iloc[:, 0].astype(float).values

categorical_params = {
    'norm': 0,                          # No normalization for categorical
    'eDim': 1,                          # No embedding needed
    'tLag': 1,
    'rescaleNorm': 1,
    'radius': 0.001,                    # Very small radius for exact matches
    'tw': 2,
    'minl': 2,
    'plotMode': 'rp',
    'pointSize': 2,
    'saveFig': False,
    'showMetrics': True,
    'doStatsFile': False
}

td, rs, mats, err_code = autoRQA(categories, categorical_params)
```

### Cross-RQA (Two Variables):
```python
from crossRQA import crossRQA

# Example 1: Rocking chair coordination
data = pd.read_csv('exampleData/RockingChairData.csv', header=None)
chair1 = data.iloc[:, 0].values
chair2 = data.iloc[:, 1].values

cross_params = {
    'norm': 1,
    'eDim': 3,
    'tLag': 15,
    'rescaleNorm': 1,
    'radius': 0.1,
    'tw': 2,                            # Not used in cross-RQA
    'minl': 2,
    'plotMode': 'rp-timeseries',
    'pointSize': 2,
    'saveFig': False,
    'showMetrics': True,
    'doStatsFile': False
}

td, rs, mats, err_code = crossRQA(chair1, chair2, cross_params)

# Example 2: Postural sway coordination (x vs y)
posture_data = pd.read_csv('exampleData/PostureData.csv', header=None)
x_sway = posture_data.iloc[:, 0].values
y_sway = posture_data.iloc[:, 1].values

td, rs, mats, err_code = crossRQA(x_sway, y_sway, cross_params)
```

### Multivariate RQA (Multiple Dimensions):
```python
from multivariateRQA import multivariateRQA, multivariateXRQA

# Example 1: 3D Lorenz system analysis
data = pd.read_csv('exampleData/lorenz_chaotic_xyz.csv')
lorenz_3d = data[['x', 'y', 'z']].values

multivar_params = {
    'norm': 'zscore',                   # Z-score normalization
    'rescaleNorm': 1,
    'radius': 0.15,
    'tw': 1,                            # Theiler window for auto-RQA
    'minl': 2,
    'plotMode': 'rp',
    'pointSize': 1,
    'saveFig': False,
    'showMetrics': True,
    'doStatsFile': False
}

# No embedding parameters needed - uses actual dimensions directly!
td, rs, mats, err_code = multivariateRQA(lorenz_3d, multivar_params)

# Example 2: 2D Postural sway analysis
posture_data = pd.read_csv('exampleData/PostureData.csv', header=None)
posture_2d = posture_data.values  # Use both x,y coordinates

td, rs, mats, err_code = multivariateRQA(posture_2d, multivar_params)

# Example 3: Multivariate cross-RQA (comparing two 3D systems)
lorenz1 = pd.read_csv('exampleData/lorenz_chaotic_xyz.csv')[['x', 'y', 'z']].values
lorenz2 = pd.read_csv('exampleData/lorenz_chaotic_xyz_2.csv')[['x', 'y', 'z']].values

td, rs, mats, err_code = multivariateXRQA(lorenz1, lorenz2, multivar_params)
```

### Diagonal Recurrence Profiles:
```python
import pandas as pd
from diagonalRP import DRP, crossDRP

# Example 1: Auto DRP (assessing % recurrence across various lags)
data = pd.read_csv('exampleData/PostureData.csv', header=None).iloc[:, 0].values

params = {
    'norm': 1,                          # Normalize to unit interval
    'eDim': 3,                          # Embedding dimension
    'tLag': 15,                         # Time lag
    'rescaleNorm': 1,                   # Rescale using mean distance
    'radius': 0.1,                      # Recurrence radius
    'tw': 1,                            # Theiler window for auto-RQA
    'maxLag': 2000,                     # Maximum lag for DRP (auto = full time  series)
    'plotMode': 'drp',                  # Plot DRP
    'pointSize': 2,                     # Size of points in the plot
    'saveFig': False,                   # Save figure
    'showMetrics': True,                # Show metrics in the console
    'doStatsFile': False                # Write statistics to file
}

drp, lags = DRP(data, params)

# Example 2: Cross DRP (good for assessing leader-follower behaviours)
data = pd.read_csv('exampleData/RockingChairData.csv', header=None)
data1 = data.iloc[:, 0].values  # First time series
data2 = data.iloc[:, 1].values  # Second time series

drp, lags = crossDRP(data1, data2, params)
```

## Data Selection Guidelines:

- **Physiological data** (PostureData, RockingChairData): Use moderate radius (0.05-0.15), embedding dimension 3-5
- **Categorical data** (Elvis): Use minimal radius (~0.001), no embedding (eDim=1, tLag=1)
- **Chaotic systems** (Lorenz): Ideal for multivariate RQA, radius 0.1-0.2
- **White noise** (WhiteNoiseData): Control/comparison data, expect low determinism
- **Short vs long series**: Longer series (>500 points) generally give more reliable results

## Analysis Tips:

1. **Start with multivariate RQA** when you have multiple related variables
2. **Use cross-RQA** to study coordination between two systems/signals  
3. **Traditional RQA** when you have a single variable or need time-delay embedding
4. **Use diagonal recurrence profiles** to assess lead/lag or leader/follower dynamics
4. **Parameter tuning**: Start with suggested values, then optimize based on your data characteristics
5. **Visual inspection**: Always examine recurrence plots to understand the underlying dynamics

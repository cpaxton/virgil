# Language Space Analysis Notebook

This document explains how to run and use the `language_space_analysis.ipynb` notebook, which explores how spatial concepts like "left" and "right" are represented in language embedding space compared to color concepts.

## Quick Start

**Quick command to run JupyterLab over network:**
```bash
cd /home/cpaxton/src/virgil
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Then access from another machine at `http://YOUR_IP:8888` using the token shown in the terminal.

## Overview

The notebook uses SigLIP (a vision-language model) to generate text embeddings for different concepts and visualizes their relationships using:

1. **PCA Scatter Plot**: Reduces high-dimensional embeddings to 2D for visualization
2. **Distance Bar Plot**: Shows distances from each concept to "left" and "right"
3. **Pairwise Distance Matrix**: Complete heatmap of all concept relationships

## Prerequisites

Before running the notebook, ensure you have:

1. **Python 3.10+** installed
2. **All dependencies** installed (see installation instructions below)
3. **CUDA-capable GPU** (recommended, but CPU will work, just slower)

## Installation

### Step 1: Install Project Dependencies

#### Option 1: Using the project's dependency management

If you're working within the virgil project:

```bash
# Install dependencies (if using uv)
uv sync

# Or if using pip
pip install -e .
```

#### Option 2: Manual installation

If you need to install dependencies manually:

```bash
pip install torch transformers numpy matplotlib seaborn scikit-learn
```

### Step 2: Install Jupyter

Jupyter is required to run the notebook. If you get a "jupyter not found" error, install it:

#### Using Conda (Recommended)

If you're using a conda environment (e.g., `virgil`):

```bash
# Activate your conda environment first
conda activate virgil

# Install Jupyter and JupyterLab
conda install -c conda-forge jupyter jupyterlab

# Or using pip in the conda environment
pip install jupyter jupyterlab
```

#### Using pip

```bash
pip install jupyter jupyterlab
```

#### Using the virgil conda environment directly

If conda activate doesn't work in your shell, you can use the full path:

```bash
# Install Jupyter in the virgil conda environment
/home/cpaxton/miniforge3/envs/virgil/bin/pip install jupyter jupyterlab
```

Then use the full path to run Jupyter:

```bash
# For local access
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab

# For network access
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

**Note:** Replace `/home/cpaxton/miniforge3/envs/virgil` with your actual conda environment path if different.

## Running the Notebook

### Method 1: Jupyter Notebook (Local)

1. **Start Jupyter Notebook**:
   ```bash
   cd /home/cpaxton/src/virgil
   jupyter notebook
   ```

2. **Navigate to the notebook**:
   - Open `scripts/language_space_analysis.ipynb`

3. **Run all cells**:
   - Use `Cell > Run All` from the menu, or
   - Press `Shift + Enter` in each cell sequentially

### Method 2: JupyterLab (Local)

1. **Start JupyterLab**:
   ```bash
   cd /home/cpaxton/src/virgil
   jupyter lab
   ```

2. **Open and run the notebook** as above

### Method 3: Jupyter Over Network

To access Jupyter from another machine on your network:

#### Using JupyterLab (Recommended)

```bash
cd /home/cpaxton/src/virgil
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

#### Using Jupyter Notebook

```bash
cd /home/cpaxton/src/virgil
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

#### Using Conda Environment Directly

If `jupyter` command is not found, use the full path:

```bash
cd /home/cpaxton/src/virgil

# For JupyterLab
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# For Jupyter Notebook
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

**Note:** The path `/home/cpaxton/miniforge3/envs/virgil` is specific to this setup. Adjust it to match your conda environment location.

#### Accessing from Another Machine

1. **Find your machine's IP address**:
   ```bash
   hostname -I
   # or
   ip addr show
   # or
   ip a
   ```
   Look for an IP address like `192.168.x.x` or `10.x.x.x` (not `127.0.0.1` which is localhost)

2. **Open in browser** from another machine:
   ```
   http://YOUR_IP_ADDRESS:8888
   ```
   For example: `http://192.168.1.100:8888`

3. **Use the token** shown in the terminal output. The output will look like:
   ```
   http://0.0.0.0:8888/lab?token=abc123def456...
   ```
   Copy the full URL including the token, or just the token part and enter it when prompted.

4. **Set a password** (optional, more convenient):
   ```bash
   jupyter notebook password
   # or
   jupyter lab password
   ```
   Then you can access without a token (just enter the password).

#### Security Notes

- The `--ip=0.0.0.0` flag makes Jupyter accessible from any machine on your network
- For production use, consider setting a password (see above) or using SSH tunneling
- The `--no-browser` flag prevents opening a browser automatically (useful for remote servers)
- Remove `--allow-root` if you're not running as root

### Method 4: VS Code / Cursor

1. **Open the notebook** in VS Code/Cursor
2. **Install the Jupyter extension** if not already installed
3. **Select a kernel** (Python environment with virgil installed)
4. **Run cells** using the play button or `Shift + Enter`

## Understanding the Notebook

### Concepts Analyzed

The notebook compares three groups of concepts:

1. **Spatial concepts (left/right)**: The main focus - how these relate to other concepts
2. **Color concepts**: red, blue, purple, green, orange, yellow
3. **Additional spatial concepts**: up, down, forward, backward, inside, outside

### Visualizations

#### 1. PCA Scatter Plot

- **Purpose**: Shows how concepts cluster in 2D space after dimensionality reduction
- **What to look for**:
  - Do spatial concepts cluster together?
  - Are colors grouped separately from spatial concepts?
  - How do additional spatial concepts relate to left/right?

#### 2. Distance Bar Plot

- **Purpose**: Quantifies how "close" each concept is to "left" and "right"
- **What to look for**:
  - Lower bars = closer to the reference concept
  - Compare distances: are colors equidistant from left/right?
  - Are additional spatial concepts closer than colors?

#### 3. Pairwise Distance Matrix

- **Purpose**: Complete view of all relationships
- **What to look for**:
  - Darker colors = smaller distances (more similar)
  - Lighter colors = larger distances (less similar)
  - Patterns in the matrix reveal concept groupings

## Customization

### Adding New Concepts

To analyze different concepts, modify the concept lists in the notebook:

```python
# Add your concepts here
spatial_concepts = ["left", "right"]
color_concepts = ["red", "blue", "purple", "green", "orange", "yellow"]
additional_concepts = ["up", "down", "forward", "backward", "inside", "outside"]

# Or add completely new categories
new_concepts = ["hot", "cold", "fast", "slow"]
```

### Changing the Model

To use a different SigLIP model:

```python
# Use a larger model (slower but potentially more accurate)
aligner = SigLIPAligner(model_name="google/siglip-large-patch16-384")

# Or use a smaller model (faster)
aligner = SigLIPAligner(model_name="google/siglip-base-patch16-256")
```

### Adjusting Visualizations

You can customize the plots by modifying:
- Figure sizes: `plt.rcParams['figure.figsize'] = (width, height)`
- Colors: Change the `c` parameter in scatter plots
- Markers: Change the `marker` parameter (e.g., 'o', 's', '^', 'v')

## Troubleshooting

### Jupyter Not Found

If you get "jupyter: command not found":

1. **Check if Jupyter is installed**:
   ```bash
   which jupyter
   ```

2. **Install Jupyter** (see Installation section above)

3. **If using conda, activate your environment**:
   ```bash
   conda activate virgil
   ```

4. **Use full path if needed**:
   ```bash
   # Find your conda environment path
   conda env list
   # or
   conda info --envs
   
   # Then use the full path (replace with your actual path)
   /home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab
   ```

5. **Check if Jupyter is installed in your environment**:
   ```bash
   # For conda environment
   /path/to/your/env/bin/pip list | grep jupyter
   
   # If not installed, install it
   /path/to/your/env/bin/pip install jupyter jupyterlab
   ```

### Import Errors

If you get import errors:

```python
# Make sure you're in the virgil directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

Also ensure you're using the correct Python kernel in Jupyter:
- In JupyterLab: Check the kernel name in the top right
- Make sure it's using the Python environment where virgil is installed

### CUDA Out of Memory

If you run out of GPU memory:

1. Use a smaller model
2. Process concepts in batches
3. Use CPU instead (slower but works)

### Model Download Issues

If the model fails to download:

1. Check your internet connection
2. Ensure you have enough disk space (~500MB for base model)
3. Try downloading manually from Hugging Face

### Network Access Issues

If you can't access Jupyter from another machine:

1. **Check firewall settings** - port 8888 may be blocked
2. **Verify IP address** - use `hostname -I` to get the correct IP
3. **Check the token** - copy the full URL with token from the terminal output
4. **Try a different port** if 8888 is in use:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8889 --no-browser
   ```

## Expected Runtime

- **Model loading**: ~10-30 seconds (first time includes download)
- **Embedding generation**: ~1-5 seconds for all concepts
- **Visualizations**: Instant

## Interpreting Results

### Key Questions to Explore

1. **Are left/right closer to each other than to colors?**
   - Check the distance between "left" and "right" vs. distances to colors

2. **Do spatial concepts form a cluster?**
   - Look at the PCA plot - are left/right/up/down/forward/backward grouped?

3. **Are colors equidistant from left/right?**
   - Check if the bar plot shows similar distances for all colors

4. **What does this tell us about language space?**
   - Consider: Are spatial and color concepts in different "regions" of embedding space?

## Further Analysis

You can extend this notebook to:

- Compare more concept categories (emotions, actions, objects, etc.)
- Use different embedding models (CLIP, sentence-transformers, etc.)
- Analyze higher-dimensional relationships (3D PCA, t-SNE, UMAP)
- Compute statistical tests on distances
- Create interactive visualizations with plotly

## Related Tools

The notebook uses utilities from `virgil.utils.embeddings`:

- `compute_pca()`: Principal Component Analysis
- `compute_distances_to_reference()`: Distance calculations
- `compute_pairwise_distances()`: Full distance matrix

These can be imported and used in other scripts:

```python
from virgil.utils.embeddings import compute_pca, compute_distances_to_reference
```

## Command Reference

### Quick Commands Cheat Sheet

**Install Jupyter in conda environment:**
```bash
/home/cpaxton/miniforge3/envs/virgil/bin/pip install jupyter jupyterlab
```

**Run JupyterLab locally:**
```bash
cd /home/cpaxton/src/virgil
jupyter lab
# or with full path:
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab
```

**Run JupyterLab over network:**
```bash
cd /home/cpaxton/src/virgil
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# or with full path:
/home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

**Find your IP address:**
```bash
hostname -I
```

**Set Jupyter password (optional):**
```bash
jupyter lab password
```

**Check if Jupyter is installed:**
```bash
which jupyter
# or
/home/cpaxton/miniforge3/envs/virgil/bin/pip list | grep jupyter
```

**List conda environments:**
```bash
conda env list
# or
conda info --envs
```

### Common Workflow

1. **First time setup:**
   ```bash
   # Install dependencies
   cd /home/cpaxton/src/virgil
   pip install -e .
   
   # Install Jupyter
   /home/cpaxton/miniforge3/envs/virgil/bin/pip install jupyter jupyterlab
   ```

2. **Run notebook over network:**
   ```bash
   cd /home/cpaxton/src/virgil
   /home/cpaxton/miniforge3/envs/virgil/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```

3. **Access from another machine:**
   - Get IP: `hostname -I`
   - Open: `http://YOUR_IP:8888`
   - Use token from terminal output

## Support

For issues or questions:
1. Check the main [README.md](../README.md)
2. Review other documentation in the `docs/` directory
3. Check the code comments in `virgil/utils/embeddings.py` and `virgil/image/siglip.py`

# Cerebras AI Dataset: Quantum Keyboard Metrics

This repository contains a script to generate the Cerebras AI Dataset: Quantum Keyboard Metrics, a comprehensive dataset designed to facilitate the development and deployment of AI models across various applications.

## Dataset Description

The Cerebras AI Dataset: Quantum Keyboard Metrics focuses on quantum-inspired keyboard metrics, including:

- **Quantum Keyboard Layouts**: 3D positions of keys for various keyboard layouts (QWERTY, Dvorak, Colemak, AZERTY, QWERTZ).
- **Word Metrics**: Distance, angle, curvature, torsion, planarity, and compactness for a large corpus of words.
- **Hex Codes**: 2D projections of 3D paths in hex code format for efficient storage and transmission.
- **Compression Ratios**: zlib and gzip compression ratios for each word's 3D path.

## Dataset Structure

The dataset consists of the following files:

1. **keyboard_layouts.json**: Dictionary of keyboard layouts with 3D key positions.
2. **word_metrics.csv**: CSV file containing word metrics (distance, angle, curvature, etc.) for each word.
3. **hex_codes.json**: Dictionary of hex codes for 2D projections of 3D paths.
4. **compression_ratios.json**: Dictionary of compression ratios for each word's 3D path.

## Dataset Generation

The dataset is generated using the `cerebras_dataset_generator.py` script, which:

1. Creates comprehensive keyboard layouts for all 5 keyboard variants
2. Processes a corpus of words (default is a sample of common words)
3. Generates metrics, hex codes, and compression data for each word-layout combination
4. Saves the results to the dataset files

### Usage

To generate the dataset, simply run:

```bash
python cerebras_dataset_generator.py
```

By default, the script uses a sample corpus of common words. To use a custom corpus, modify the `load_word_corpus` function to load words from a file.

## Dataset Integration and Deployment

To integrate and deploy this dataset across all AI models at Cerebras, follow these steps:

1. **Data Preprocessing**: Clean and preprocess the dataset to ensure consistency and compatibility with AI models.
2. **Data Storage**: Store the dataset in a scalable and accessible storage solution, such as a cloud-based object storage or a data lake.
3. **Model Integration**: Integrate the dataset with AI models using APIs or data pipelines.
4. **Model Deployment**: Deploy AI models with the integrated dataset using Cerebras' AI platform.

## Example Code

Here's an example code snippet in Python to load and use the dataset:

```python
import pandas as pd
import json

# Load keyboard layouts
keyboard_layouts = json.load(open('keyboard_layouts.json'))

# Load word metrics
word_metrics = pd.read_csv('word_metrics.csv')

# Load hex codes
hex_codes = json.load(open('hex_codes.json'))

# Load compression ratios
compression_ratios = json.load(open('compression_ratios.json'))

# Use the dataset to train an AI model
# model = train_model(word_metrics, keyboard_layouts, hex_codes, compression_ratios)
```

## Benefits

The Cerebras AI Dataset: Quantum Keyboard Metrics provides several benefits:

1. **Comprehensive Data**: The dataset includes a wide range of metrics and data points for various keyboard layouts and words.
2. **Standardized Format**: The dataset is provided in standardized formats (JSON and CSV) for easy integration with AI models.
3. **Quantum-Inspired Metrics**: The dataset includes quantum-inspired metrics that can be used to develop more sophisticated AI models.
4. **Efficient Storage**: The dataset includes compression ratios and hex codes for efficient storage and transmission of 3D paths.

## Implementation Details

The `cerebras_dataset_generator.py` script is a standalone Python script that doesn't require any external dependencies beyond the standard library. It includes functions to:

- Create comprehensive keyboard layouts for all 5 keyboard variants
- Load a corpus of words for analysis
- Generate sample metrics, hex codes, and compression data for each word-layout combination
- Save the results to the dataset files

The script uses random number generation to create sample data that mimics the expected patterns and relationships between words, layouts, and metrics. In a real-world implementation, these metrics would be calculated using more sophisticated algorithms based on the actual 3D paths of words on different keyboard layouts.

## Quantum Enhanced Keyboard

The repository now includes `quantum_enhanced_keyboard.py`, a sophisticated implementation of a quantum-inspired keyboard analysis system. This script provides a more advanced and comprehensive approach to keyboard analysis compared to the dataset generator.

### Features

- Multiple keyboard layout support (QWERTY, Dvorak, Colemak)
- Quantum-inspired metrics for typing analysis
- 3D visualization of typing paths
- Statistical analysis with confidence intervals
- Machine learning integration (PCA, K-means clustering)
- Path compression and hex code generation

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the main script to see a demonstration of the quantum keyboard analysis:

```bash
python quantum_enhanced_keyboard.py
```

### Example Code

```python
from quantum_enhanced_keyboard import QuantumKeyboard

# Create a keyboard with QWERTY layout
keyboard = QuantumKeyboard('qwerty')

# Analyze a word
analysis = keyboard.analyze_word("hello")

# Print metrics
print(analysis['metrics'])

# Get quantum typing statistics
stats = keyboard.calculate_comprehensive_quantum_stats("hello")
print(f"Quantum Coherence: {stats.quantum_coherence}")
print(f"Harmonic Resonance: {stats.harmonic_resonance}")

# Visualize the word path
keyboard.plot_quantum_analysis("hello")
```

### Advanced Usage

Compare multiple keyboard layouts:

```python
from quantum_enhanced_keyboard import compare_quantum_keyboards

words = ["hello", "world", "quantum", "keyboard"]
layouts = ["qwerty", "dvorak", "colemak"]

results = compare_quantum_keyboards(words, layouts)
```

Export analysis results to JSON:

```python
from quantum_enhanced_keyboard import export_analysis_results

export_analysis_results(results, "analysis_results.json")
```

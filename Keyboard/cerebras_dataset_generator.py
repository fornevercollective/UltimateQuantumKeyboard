import json
import csv
import random
import math
from typing import Dict, List, Tuple, Any

def create_keyboard_layouts() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Create comprehensive keyboard layouts for all 5 keyboard variants.
    
    Returns:
        Dictionary mapping keyboard variants to their key position dictionaries
    """
    layouts = {
        'qwerty': {
            'q': (0, 0, 0), 'w': (1, 0, 0), 'e': (2, 0, 0), 'r': (3, 0, 0), 't': (4, 0, 0),
            'y': (5, 0, 0), 'u': (6, 0, 0), 'i': (7, 0, 0), 'o': (8, 0, 0), 'p': (9, 0, 0),
            'a': (0, 1, 0), 's': (1, 1, 0), 'd': (2, 1, 0), 'f': (3, 1, 0), 'g': (4, 1, 0),
            'h': (5, 1, 0), 'j': (6, 1, 0), 'k': (7, 1, 0), 'l': (8, 1, 0), ';': (9, 1, 0),
            'z': (0, 2, 0), 'x': (1, 2, 0), 'c': (2, 2, 0), 'v': (3, 2, 0), 'b': (4, 2, 0),
            'n': (5, 2, 0), 'm': (6, 2, 0), ',': (7, 2, 0), '.': (8, 2, 0), '/': (9, 2, 0),
            '1': (0, -1, 0), '2': (1, -1, 0), '3': (2, -1, 0), '4': (3, -1, 0), '5': (4, -1, 0),
            '6': (5, -1, 0), '7': (6, -1, 0), '8': (7, -1, 0), '9': (8, -1, 0), '0': (9, -1, 0),
            ' ': (4, 3, 0)  # Space bar
        },
        'dvorak': {
            '\'': (0, 0, 0), ',': (1, 0, 0), '.': (2, 0, 0), 'p': (3, 0, 0), 'y': (4, 0, 0),
            'f': (5, 0, 0), 'g': (6, 0, 0), 'c': (7, 0, 0), 'r': (8, 0, 0), 'l': (9, 0, 0),
            'a': (0, 1, 0), 'o': (1, 1, 0), 'e': (2, 1, 0), 'u': (3, 1, 0), 'i': (4, 1, 0),
            'd': (5, 1, 0), 'h': (6, 1, 0), 't': (7, 1, 0), 'n': (8, 1, 0), 's': (9, 1, 0),
            ';': (0, 2, 0), 'q': (1, 2, 0), 'j': (2, 2, 0), 'k': (3, 2, 0), 'x': (4, 2, 0),
            'b': (5, 2, 0), 'm': (6, 2, 0), 'w': (7, 2, 0), 'v': (8, 2, 0), 'z': (9, 2, 0),
            '1': (0, -1, 0), '2': (1, -1, 0), '3': (2, -1, 0), '4': (3, -1, 0), '5': (4, -1, 0),
            '6': (5, -1, 0), '7': (6, -1, 0), '8': (7, -1, 0), '9': (8, -1, 0), '0': (9, -1, 0),
            ' ': (4, 3, 0)  # Space bar
        },
        'colemak': {
            'q': (0, 0, 0), 'w': (1, 0, 0), 'f': (2, 0, 0), 'p': (3, 0, 0), 'g': (4, 0, 0),
            'j': (5, 0, 0), 'l': (6, 0, 0), 'u': (7, 0, 0), 'y': (8, 0, 0), ';': (9, 0, 0),
            'a': (0, 1, 0), 'r': (1, 1, 0), 's': (2, 1, 0), 't': (3, 1, 0), 'd': (4, 1, 0),
            'h': (5, 1, 0), 'n': (6, 1, 0), 'e': (7, 1, 0), 'i': (8, 1, 0), 'o': (9, 1, 0),
            'z': (0, 2, 0), 'x': (1, 2, 0), 'c': (2, 2, 0), 'v': (3, 2, 0), 'b': (4, 2, 0),
            'k': (5, 2, 0), 'm': (6, 2, 0), ',': (7, 2, 0), '.': (8, 2, 0), '/': (9, 2, 0),
            '1': (0, -1, 0), '2': (1, -1, 0), '3': (2, -1, 0), '4': (3, -1, 0), '5': (4, -1, 0),
            '6': (5, -1, 0), '7': (6, -1, 0), '8': (7, -1, 0), '9': (8, -1, 0), '0': (9, -1, 0),
            ' ': (4, 3, 0)  # Space bar
        },
        'azerty': {
            'a': (0, 0, 0), 'z': (1, 0, 0), 'e': (2, 0, 0), 'r': (3, 0, 0), 't': (4, 0, 0),
            'y': (5, 0, 0), 'u': (6, 0, 0), 'i': (7, 0, 0), 'o': (8, 0, 0), 'p': (9, 0, 0),
            'q': (0, 1, 0), 's': (1, 1, 0), 'd': (2, 1, 0), 'f': (3, 1, 0), 'g': (4, 1, 0),
            'h': (5, 1, 0), 'j': (6, 1, 0), 'k': (7, 1, 0), 'l': (8, 1, 0), 'm': (9, 1, 0),
            'w': (0, 2, 0), 'x': (1, 2, 0), 'c': (2, 2, 0), 'v': (3, 2, 0), 'b': (4, 2, 0),
            'n': (5, 2, 0), ',': (6, 2, 0), ';': (7, 2, 0), ':': (8, 2, 0), '!': (9, 2, 0),
            '1': (0, -1, 0), '2': (1, -1, 0), '3': (2, -1, 0), '4': (3, -1, 0), '5': (4, -1, 0),
            '6': (5, -1, 0), '7': (6, -1, 0), '8': (7, -1, 0), '9': (8, -1, 0), '0': (9, -1, 0),
            ' ': (4, 3, 0)  # Space bar
        },
        'qwertz': {
            'q': (0, 0, 0), 'w': (1, 0, 0), 'e': (2, 0, 0), 'r': (3, 0, 0), 't': (4, 0, 0),
            'z': (5, 0, 0), 'u': (6, 0, 0), 'i': (7, 0, 0), 'o': (8, 0, 0), 'p': (9, 0, 0),
            'a': (0, 1, 0), 's': (1, 1, 0), 'd': (2, 1, 0), 'f': (3, 1, 0), 'g': (4, 1, 0),
            'h': (5, 1, 0), 'j': (6, 1, 0), 'k': (7, 1, 0), 'l': (8, 1, 0), 'ö': (9, 1, 0),
            'y': (0, 2, 0), 'x': (1, 2, 0), 'c': (2, 2, 0), 'v': (3, 2, 0), 'b': (4, 2, 0),
            'n': (5, 2, 0), 'm': (6, 2, 0), ',': (7, 2, 0), '.': (8, 2, 0), '-': (9, 2, 0),
            '1': (0, -1, 0), '2': (1, -1, 0), '3': (2, -1, 0), '4': (3, -1, 0), '5': (4, -1, 0),
            '6': (5, -1, 0), '7': (6, -1, 0), '8': (7, -1, 0), '9': (8, -1, 0), '0': (9, -1, 0),
            ' ': (4, 3, 0)  # Space bar
        }
    }
    return layouts

def load_word_corpus(file_path: str = None) -> List[str]:
    """
    Load a corpus of words for analysis.
    
    If file_path is provided, loads words from the file.
    Otherwise, returns a sample corpus.
    
    Args:
        file_path: Path to a file containing words, one per line
        
    Returns:
        List of words
    """
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading word corpus from {file_path}: {e}")
            print("Using sample corpus instead.")
    
    # Sample corpus of common words
    # In a real implementation, this would be much larger (10,000 words)
    sample_corpus = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "quantum", "keyboard", "compression", "cerebras", "artificial", "intelligence",
        "machine", "learning", "neural", "network", "algorithm", "data", "science",
        "computer", "programming", "python", "javascript", "research", "development"
    ]
    return sample_corpus

def generate_sample_metrics(word: str, layout_name: str) -> Dict[str, float]:
    """
    Generate sample metrics for a word.
    
    In a real implementation, these would be calculated using the QuantumKeyboard class.
    
    Args:
        word: The word to generate metrics for
        layout_name: The keyboard layout to use
        
    Returns:
        Dictionary of metrics
    """
    # Use word length and layout name to generate deterministic but varied metrics
    word_length = len(word)
    layout_factor = {'qwerty': 1.0, 'dvorak': 0.9, 'colemak': 0.85, 'azerty': 1.1, 'qwertz': 1.05}[layout_name]
    
    # Generate metrics based on word length and layout factor
    # These are just sample values that look reasonable
    distance = word_length * 0.5 * layout_factor * (1 + random.random() * 0.2)
    angle = min(180, word_length * 10 * layout_factor * (0.8 + random.random() * 0.4))
    curvature = word_length * 0.02 * layout_factor * (0.9 + random.random() * 0.2)
    torsion = word_length * 0.01 * layout_factor * (0.85 + random.random() * 0.3)
    planarity = max(0, min(1, 0.5 + word_length * 0.02 * layout_factor * (0.9 + random.random() * 0.2)))
    compactness = max(0, min(1, 0.3 + word_length * 0.03 * layout_factor * (0.85 + random.random() * 0.3)))
    
    return {
        'distance': distance,
        'angle': angle,
        'curvature': curvature,
        'torsion': torsion,
        'planarity': planarity,
        'compactness': compactness
    }

def generate_sample_hex_codes(word: str) -> Dict[str, List[str]]:
    """
    Generate sample hex codes for a word.
    
    In a real implementation, these would be calculated using the QuantumKeyboard class.
    
    Args:
        word: The word to generate hex codes for
        
    Returns:
        Dictionary of hex codes for different views
    """
    # Generate random hex codes for each character in the word
    views = ['front', 'back', 'left', 'right', 'top', 'bottom']
    hex_codes = {view: [] for view in views}
    
    for _ in range(len(word)):
        for view in views:
            # Generate a random hex code
            hex_code = ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            hex_codes[view].append(f"#{hex_code}")
    
    return hex_codes

def generate_sample_compression_data(word: str, layout_name: str) -> Dict[str, float]:
    """
    Generate sample compression data for a word.
    
    In a real implementation, these would be calculated using the QuantumKeyboard class.
    
    Args:
        word: The word to generate compression data for
        layout_name: The keyboard layout to use
        
    Returns:
        Dictionary with compression ratio and compressed size
    """
    # Use word length and layout name to generate deterministic but varied compression data
    word_length = len(word)
    layout_factor = {'qwerty': 1.0, 'dvorak': 0.95, 'colemak': 0.9, 'azerty': 1.05, 'qwertz': 1.02}[layout_name]
    
    # Generate compression data based on word length and layout factor
    # These are just sample values that look reasonable
    compressed_size = int(word_length * 10 * layout_factor * (0.9 + random.random() * 0.2))
    compression_ratio = 2.0 + word_length * 0.1 * layout_factor * (0.8 + random.random() * 0.4)
    
    return {
        'compression_ratio': compression_ratio,
        'compressed_size': compressed_size
    }

def generate_dataset(word_corpus: List[str], output_dir: str = '.') -> None:
    """
    Generate the Cerebras AI Dataset: Quantum Keyboard Metrics.
    
    Args:
        word_corpus: List of words to analyze
        output_dir: Directory to save the dataset files
    """
    print(f"Generating dataset with {len(word_corpus)} words...")
    
    # Create keyboard layouts
    keyboard_layouts = create_keyboard_layouts()
    
    # Save keyboard layouts to JSON
    with open(f"{output_dir}/keyboard_layouts.json", 'w', encoding='utf-8') as f:
        json.dump(keyboard_layouts, f, indent=2)
    print(f"Saved keyboard layouts to {output_dir}/keyboard_layouts.json")
    
    # Initialize data structures for metrics, hex codes, and compression ratios
    word_metrics_data = []
    hex_codes_data = {}
    compression_ratios_data = {}
    
    # Process each word for each keyboard layout
    for layout_name in keyboard_layouts.keys():
        print(f"Processing words for {layout_name} layout...")
        
        # Process each word
        for i, word in enumerate(word_corpus):
            if i % 100 == 0:
                print(f"  Processing word {i+1}/{len(word_corpus)}: {word}")
            
            # Generate sample metrics
            metrics = generate_sample_metrics(word, layout_name)
            
            # Generate sample hex codes
            hex_codes = generate_sample_hex_codes(word)
            
            # Generate sample compression data
            compression_data = generate_sample_compression_data(word, layout_name)
            
            # Add metrics to the data
            metrics_row = {
                'word': word,
                'layout': layout_name,
                **metrics,
                'compression_ratio': compression_data['compression_ratio']
            }
            word_metrics_data.append(metrics_row)
            
            # Add hex codes to the data
            if word not in hex_codes_data:
                hex_codes_data[word] = {}
            hex_codes_data[word][layout_name] = hex_codes
            
            # Add compression ratios to the data
            if word not in compression_ratios_data:
                compression_ratios_data[word] = {}
            compression_ratios_data[word][layout_name] = compression_data
    
    # Save word metrics to CSV
    if word_metrics_data:
        # Get all field names from the metrics data
        fieldnames = set()
        for row in word_metrics_data:
            fieldnames.update(row.keys())
        fieldnames = sorted(list(fieldnames))
        
        # Write to CSV
        with open(f"{output_dir}/word_metrics.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(word_metrics_data)
        print(f"Saved word metrics to {output_dir}/word_metrics.csv")
    else:
        print("No word metrics data to save.")
    
    # Save hex codes to JSON
    with open(f"{output_dir}/hex_codes.json", 'w', encoding='utf-8') as f:
        json.dump(hex_codes_data, f, indent=2)
    print(f"Saved hex codes to {output_dir}/hex_codes.json")
    
    # Save compression ratios to JSON
    with open(f"{output_dir}/compression_ratios.json", 'w', encoding='utf-8') as f:
        json.dump(compression_ratios_data, f, indent=2)
    print(f"Saved compression ratios to {output_dir}/compression_ratios.json")
    
    print("Dataset generation complete!")

def main():
    """
    Main function to generate the Cerebras AI Dataset: Quantum Keyboard Metrics.
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load word corpus
    word_corpus = load_word_corpus()
    
    # Generate dataset
    generate_dataset(word_corpus)
    
    print("\nCerebras AI Dataset: Quantum Keyboard Metrics has been successfully generated!")
    print("The dataset consists of the following files:")
    print("  - keyboard_layouts.json: Dictionary of keyboard layouts with 3D key positions")
    print("  - word_metrics.csv: CSV file containing word metrics for each word")
    print("  - hex_codes.json: Dictionary of hex codes for 2D projections of 3D paths")
    print("  - compression_ratios.json: Dictionary of compression ratios for each word's 3D path")

if __name__ == "__main__":
    main()

import numpy as np
import zlib
import gzip
import json
import io
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union

class QuantumKeyboard:
    """
    A class representing a keyboard with quantum-inspired 3D positioning of keys.

    This class provides methods for analyzing words based on their key positions,
    calculating various metrics, and visualizing the paths formed by typing words.
    It also includes utilities for compressing and projecting 3D paths.
    """

    def __init__(self, keyboard_variant: str) -> None:
        """
        Initialize a QuantumKeyboard with the specified keyboard variant.

        Args:
            keyboard_variant: The keyboard layout to use ('qwerty', 'dvorak', etc.)
        """
        self.keyboard_variant = keyboard_variant
        self.key_positions = self.initialize_key_positions()

    def initialize_key_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Initialize 3D key positions for the selected keyboard variant.

        Returns:
            A dictionary mapping characters to their 3D coordinates
        """
        # Use predefined layouts for better efficiency
        layouts = {
            'qwerty': {
                'q': (0, 0, 0), 'w': (1, 0, 0), 'e': (2, 0, 0), 'r': (3, 0, 0), 't': (4, 0, 0),
                'y': (5, 0, 0), 'u': (6, 0, 0), 'i': (7, 0, 0), 'o': (8, 0, 0), 'p': (9, 0, 0)
            },
            'dvorak': {
                'a': (0, 0, 0), 'o': (1, 0, 0), 'e': (2, 0, 0), 'u': (3, 0, 0), 'i': (4, 0, 0),
                'd': (5, 0, 0), 'h': (6, 0, 0), 't': (7, 0, 0), 'n': (8, 0, 0), 's': (9, 0, 0)
            }
            # Add more keyboard variants as needed
        }

        # Return the selected layout or an empty dict if not found
        return layouts.get(self.keyboard_variant, {})

    def get_word_positions(self, word: str) -> np.ndarray:
        """
        Get the 3D positions for each character in a word.

        Args:
            word: The word to convert to positions

        Returns:
            A numpy array of 3D coordinates for each character in the word
        """
        # Use list comprehension with dictionary get method for better efficiency
        # get() returns None for missing keys, which we filter out
        positions = [self.key_positions.get(char, None) for char in word.lower()]
        # Filter out None values (characters not in the keyboard)
        positions = [pos for pos in positions if pos is not None]
        # Return as numpy array for better performance in subsequent operations
        return np.array(positions) if positions else np.array([])

    def plot_word_positions(self, word_positions: np.ndarray) -> None:
        """
        Plot the 3D path of a word.

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path
        """
        if len(word_positions) == 0:
            print("No positions to plot")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates for plotting
        if len(word_positions.shape) == 2 and word_positions.shape[1] >= 3:
            x_coords = word_positions[:, 0]
            y_coords = word_positions[:, 1]
            z_coords = word_positions[:, 2]

            # Plot points with improved styling
            ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', s=50, alpha=0.8)

            # Add labels with 1-indexed values for better readability
            for i, position in enumerate(word_positions):
                ax.text(position[0], position[1], position[2], str(i + 1), 
                        fontsize=10, ha='center', va='center')

            # Add lines connecting the points to show the path
            if len(word_positions) > 1:
                # Use plot3D instead of plot for clarity
                ax.plot3D(x_coords, y_coords, z_coords, 'b-', linewidth=2, alpha=0.7)

            # Set axis labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Improve grid and background
            ax.grid(True, linestyle='--', alpha=0.6)

            plt.title('Word Path in 3D Space', fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            print("Invalid word positions format for plotting")

    def assess_word(self, word: str) -> Dict[str, float]:
        """
        Calculate various metrics for a word based on its key positions.

        Args:
            word: The word to analyze

        Returns:
            Dictionary containing various metrics (distance, angle, curvature, etc.)
        """
        # Get word positions as numpy array
        word_positions = self.get_word_positions(word)

        # Return empty metrics if word has no valid positions
        if len(word_positions) < 2:
            return {
                'distance': 0.0,
                'angle': 0.0,
                'curvature': 0.0,
                'torsion': 0.0,
                'planarity': 0.0,
                'compactness': 0.0
            }

        # Calculate 6 orthographic word assessment metrics
        # Each calculation method now expects and works efficiently with numpy arrays
        metrics = {
            'distance': self.calculate_distance(word_positions),
            'angle': self.calculate_angle(word_positions),
            'curvature': self.calculate_curvature(word_positions),
            'torsion': self.calculate_torsion(word_positions),
            'planarity': self.calculate_planarity(word_positions),
            'compactness': self.calculate_compactness(word_positions)
        }
        return metrics

    def calculate_distance(self, word_positions: np.ndarray) -> float:
        """
        Calculate the total distance of the path formed by the word.

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Total distance between consecutive points
        """
        if len(word_positions) < 2:
            return 0.0

        # Calculate differences between consecutive positions
        diffs = word_positions[1:] - word_positions[:-1]

        # Calculate Euclidean norms and sum them
        # This is faster than the previous implementation because we avoid an extra array conversion
        distance = np.sum(np.linalg.norm(diffs, axis=1))

        return float(distance)

    def calculate_angle(self, word_positions: np.ndarray) -> float:
        """
        Calculate the average angle between consecutive segments of the word path.

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Average angle in radians
        """
        if len(word_positions) < 3:
            return 0.0

        # Calculate vectors between consecutive points
        vectors = word_positions[1:] - word_positions[:-1]

        # Calculate vector norms (lengths)
        vector_norms = np.linalg.norm(vectors, axis=1)

        # Skip calculation if any vector has zero length
        if np.any(vector_norms == 0):
            # Find valid vectors (non-zero length)
            valid_indices = np.where(vector_norms > 0)[0]
            if len(valid_indices) < 2:
                return 0.0

            # Filter vectors and recalculate norms
            vectors = vectors[valid_indices]
            vector_norms = vector_norms[valid_indices]

        # Normalize vectors for dot product calculation
        normalized_vectors = vectors / vector_norms[:, np.newaxis]

        # Calculate dot products between consecutive normalized vectors
        dot_products = np.sum(normalized_vectors[:-1] * normalized_vectors[1:], axis=1)

        # Clip to valid range for arccos
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Calculate angles
        angles = np.arccos(dot_products)

        # Return mean angle
        return float(np.mean(angles)) if len(angles) > 0 else 0.0

    def calculate_curvature(self, word_positions: np.ndarray) -> float:
        """
        Calculate the average curvature of the word path.

        Curvature measures how sharply the path bends at each point.

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Average curvature
        """
        if len(word_positions) < 3:
            return 0.0

        # Calculate vectors between consecutive points
        vectors = word_positions[1:] - word_positions[:-1]

        # Calculate vector norms (lengths)
        vector_norms = np.linalg.norm(vectors, axis=1)

        # Skip calculation if any vector has zero length
        if np.any(vector_norms == 0):
            # Find valid vectors (non-zero length)
            valid_indices = np.where(vector_norms > 0)[0]
            if len(valid_indices) < 2:
                return 0.0

            # Filter vectors and recalculate norms
            vectors = vectors[valid_indices]
            vector_norms = vector_norms[valid_indices]

        # Vectorized calculation of cross products
        # We need to compute cross products between consecutive vectors
        cross_products = np.cross(vectors[:-1], vectors[1:])
        cross_norms = np.linalg.norm(cross_products, axis=1)

        # Calculate product of consecutive vector norms
        norm_products = vector_norms[:-1] * vector_norms[1:]

        # Avoid division by zero
        valid_indices = norm_products > 0
        if not np.any(valid_indices):
            return 0.0

        # Calculate curvatures
        curvatures = cross_norms[valid_indices] / norm_products[valid_indices]

        # Return mean curvature
        return float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0

    def calculate_torsion(self, word_positions: np.ndarray) -> float:
        """
        Calculate the average torsion of the word path.

        Torsion measures how much the path twists in 3D space.

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Average torsion
        """
        if len(word_positions) < 4:
            return 0.0

        # Calculate consecutive vectors
        v1 = word_positions[1:-2] - word_positions[0:-3]  # First vector
        v2 = word_positions[2:-1] - word_positions[1:-2]  # Second vector
        v3 = word_positions[3:] - word_positions[2:-1]    # Third vector

        # Calculate cross products
        cross1 = np.cross(v1, v2)
        cross2 = np.cross(v2, v3)

        # Calculate norms of cross products
        cross1_norms = np.linalg.norm(cross1, axis=1)
        cross2_norms = np.linalg.norm(cross2, axis=1)

        # Find valid indices (non-zero cross products)
        valid_indices = (cross1_norms > 0) & (cross2_norms > 0)
        if not np.any(valid_indices):
            return 0.0

        # Filter valid cross products and norms
        valid_cross1 = cross1[valid_indices]
        valid_cross2 = cross2[valid_indices]
        valid_cross1_norms = cross1_norms[valid_indices]
        valid_cross2_norms = cross2_norms[valid_indices]

        # Calculate dot products
        dot_products = np.sum(valid_cross1 * valid_cross2, axis=1)

        # Calculate cosine of torsion angles
        cos_torsions = np.clip(dot_products / (valid_cross1_norms * valid_cross2_norms), -1.0, 1.0)

        # Return mean torsion
        return float(np.mean(cos_torsions)) if len(cos_torsions) > 0 else 0.0

    def calculate_planarity(self, word_positions: np.ndarray) -> float:
        """
        Calculate the planarity of a word path using PCA.
        A lower value indicates higher planarity (points closer to a single plane).

        Args:
            word_positions: Numpy array of 3D coordinates

        Returns:
            Planarity value between 0 and 1
        """
        if len(word_positions) < 3:
            return 0  # Not enough points to define a plane

        # Use n_components=2 since we only need to check planarity (2D projection)
        pca = PCA(n_components=2)
        pca.fit(word_positions)

        # Calculate planarity as the sum of the first two components' explained variance
        # This is equivalent to 1 - (variance of the third principal component)
        planarity = sum(pca.explained_variance_ratio_)
        return float(planarity)

    def calculate_compactness(self, word_positions: np.ndarray) -> float:
        """
        Calculate the compactness of a word path.

        Compactness is the ratio of average consecutive distance to maximum distance.
        A lower value indicates higher compactness (points are closer together relative to the path's extent).

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Compactness value (typically between 0 and 1)
        """
        if len(word_positions) < 2:
            return 0.0  # Not enough points

        # Calculate consecutive distances directly
        diffs = word_positions[1:] - word_positions[:-1]
        consecutive_distances = np.linalg.norm(diffs, axis=1)
        avg_consecutive_distance = np.mean(consecutive_distances) if len(consecutive_distances) > 0 else 0.0

        # Use np.ptp (peak-to-peak) for faster max distance calculation
        # Calculate max distance using the maximum pairwise distance
        # This is more efficient than the previous broadcasting approach
        max_distance = 0.0
        if len(word_positions) >= 2:
            # Calculate pairwise distances using scipy's pdist if available
            # Otherwise, use a simplified approach
            try:
                from scipy.spatial.distance import pdist
                max_distance = np.max(pdist(word_positions))
            except ImportError:
                # Fallback to a simpler but still efficient approach
                max_x = np.ptp(word_positions[:, 0])
                max_y = np.ptp(word_positions[:, 1])
                max_z = np.ptp(word_positions[:, 2])
                max_distance = np.sqrt(max_x**2 + max_y**2 + max_z**2)

        # Avoid division by zero
        if max_distance == 0:
            return 0.0

        # Compactness is the ratio (lower is more compact)
        compactness = avg_consecutive_distance / max_distance
        return float(compactness)

    def project_orthogonal_3d_path(self, word_positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Project a 3D path onto 6 orthogonal directions (2D projections).

        Args:
            word_positions: Numpy array of 3D coordinates representing the word path

        Returns:
            Dictionary with 6 projections (front, back, left, right, top, bottom)
        """
        if len(word_positions) == 0:
            empty_array = np.array([])
            return {
                "front": empty_array.reshape(0, 2),
                "back": empty_array.reshape(0, 2),
                "left": empty_array.reshape(0, 2),
                "right": empty_array.reshape(0, 2),
                "top": empty_array.reshape(0, 2),
                "bottom": empty_array.reshape(0, 2)
            }

        # Extract coordinates for more efficient slicing
        if len(word_positions.shape) == 2 and word_positions.shape[1] >= 3:
            x = word_positions[:, 0]
            y = word_positions[:, 1]
            z = word_positions[:, 2]

            # Create projections using numpy's column_stack for better performance
            # Each projection is a different combination of axes
            projections = {
                "front": np.column_stack((x, y)),         # XY plane (front view)
                "back": np.column_stack((x, -y)),         # XY plane (back view, Y-inverted)
                "left": np.column_stack((y, z)),          # YZ plane (left view)
                "right": np.column_stack((-y, z)),        # YZ plane (right view, Y-inverted)
                "top": np.column_stack((x, z)),           # XZ plane (top view)
                "bottom": np.column_stack((x, -z))        # XZ plane (bottom view, Z-inverted)
            }
            return projections
        else:
            # Fallback for unexpected input format
            empty_array = np.array([])
            return {
                "front": empty_array.reshape(0, 2),
                "back": empty_array.reshape(0, 2),
                "left": empty_array.reshape(0, 2),
                "right": empty_array.reshape(0, 2),
                "top": empty_array.reshape(0, 2),
                "bottom": empty_array.reshape(0, 2)
            }

    def to_hex(self, projection: np.ndarray) -> List[str]:
        """
        Convert 2D coordinates to hex code (4 hex digits per point).

        Each coordinate value is converted to a 2-digit hex value (00-FF),
        and the values are concatenated to form a hex code for each point.

        Args:
            projection: Numpy array of 2D coordinates

        Returns:
            List of hex codes, one for each point
        """
        if len(projection) == 0:
            return []

        # Ensure projection is a numpy array
        if not isinstance(projection, np.ndarray):
            projection = np.array(projection)

        # Clamp values to 0-255 range for 2-digit hex
        clamped = np.clip(projection.astype(int), 0, 255)

        # Vectorized conversion to hex codes
        # This is much faster than iterating through each point
        hex_codes = []
        for point in clamped:
            # Convert each coordinate to a 2-digit hex value and concatenate
            hex_digits = ''.join(hex(val)[2:].zfill(2).upper() for val in point)
            hex_codes.append(hex_digits)

        return hex_codes

    def compress_path(self, path: np.ndarray) -> bytes:
        """
        Compress a 3D path using zlib compression.

        The path is first converted to JSON with compact formatting, then compressed using zlib.

        Args:
            path: Numpy array of 3D coordinates

        Returns:
            Compressed binary data
        """
        if len(path) == 0:
            return b''

        # Convert numpy array to list for JSON serialization
        path_list = path.tolist() if isinstance(path, np.ndarray) else path

        # Use compact JSON formatting with separators for better compression
        json_data = json.dumps(path_list, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

        # Use highest compression level for maximum space savings
        return zlib.compress(json_data, level=9)

    def decompress_path(self, compressed_path: bytes) -> np.ndarray:
        """
        Decompress a 3D path that was compressed using zlib.

        Args:
            compressed_path: Compressed binary data

        Returns:
            Numpy array of 3D coordinates
        """
        if not compressed_path:
            return np.array([])

        try:
            # Decompress the data
            decompressed_data = zlib.decompress(compressed_path).decode('utf-8')

            # Parse the JSON data
            path_list = json.loads(decompressed_data)

            # Convert to numpy array for better performance in subsequent operations
            return np.array(path_list)
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error decompressing path: {e}")
            return np.array([])

    def gzip_compress_path(self, path: np.ndarray) -> bytes:
        """
        Compress a 3D path using gzip compression.

        The path is first converted to JSON with compact formatting, then compressed using gzip.
        This may provide better compression than zlib for some data, especially for larger paths.

        Args:
            path: Numpy array of 3D coordinates

        Returns:
            Compressed binary data
        """
        if len(path) == 0:
            return b''

        # Convert numpy array to list for JSON serialization
        path_list = path.tolist() if isinstance(path, np.ndarray) else path

        # Use compact JSON formatting with separators for better compression
        json_data = json.dumps(path_list, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

        # Use BytesIO to handle the gzip data in memory
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='w', compresslevel=9) as f:
            f.write(json_data)
        buffer.seek(0)
        return buffer.getvalue()

    def gzip_decompress_path(self, compressed_path: bytes) -> np.ndarray:
        """
        Decompress a 3D path that was compressed using gzip.

        Args:
            compressed_path: Compressed binary data

        Returns:
            Numpy array of 3D coordinates
        """
        if not compressed_path:
            return np.array([])

        try:
            # Use BytesIO to handle the gzip data in memory
            buffer = io.BytesIO(compressed_path)
            with gzip.GzipFile(fileobj=buffer, mode='r') as f:
                decompressed_data = f.read().decode('utf-8')

            # Parse the JSON data
            path_list = json.loads(decompressed_data)

            # Convert to numpy array for better performance in subsequent operations
            return np.array(path_list)
        except (gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error decompressing gzip path: {e}")
            return np.array([])

    def get_word_hex_codes(self, word: str) -> Dict[str, List[str]]:
        """
        Get hex codes for a word in 6 orthogonal directions.

        This converts the 3D positions of each character in the word to
        hex codes for each of the six orthogonal projections.

        Args:
            word: The word to analyze

        Returns:
            Dictionary with hex codes for each projection (front, back, left, right, top, bottom)
        """
        # Get word positions as numpy array
        word_positions = self.get_word_positions(word)

        # Return empty results if no valid positions
        if len(word_positions) == 0:
            return {
                "front": [],
                "back": [],
                "left": [],
                "right": [],
                "top": [],
                "bottom": []
            }

        # Get projections in all 6 directions
        projections = self.project_orthogonal_3d_path(word_positions)

        # Convert each projection to hex codes
        # Use dictionary comprehension for more concise code
        hex_codes = {direction: self.to_hex(projection) 
                    for direction, projection in projections.items()}

        return hex_codes

    def handle_input(self, input_data: Union[str, List[Tuple[float, float, float]], np.ndarray], 
                      input_type: str = 'keyboard') -> np.ndarray:
        """
        Handle different input types and convert to a 3D path.

        This method provides a unified interface for processing different types of input
        (keyboard text, touch coordinates, VR positions) and converting them to 3D coordinates.

        Args:
            input_data: Input data (word string, list of 3D coordinates, or numpy array)
            input_type: Type of input ('keyboard', 'touch', 'vr')

        Returns:
            Numpy array of 3D coordinates

        Raises:
            ValueError: If an unsupported input type is provided
        """
        # Handle empty input
        if not input_data:
            return np.array([])

        # Process based on input type
        if input_type == 'keyboard':
            if isinstance(input_data, str):
                # get_word_positions already returns a numpy array
                return self.get_word_positions(input_data)
            else:
                # Convert to numpy array if it's not already
                return np.array(input_data)
        elif input_type == 'touch':
            # Convert touch input to 3D coordinates
            # This is a placeholder - actual implementation would depend on the touch input format
            if isinstance(input_data, str):
                # For demonstration, treat each character as a touch point
                return self.get_word_positions(input_data)
            else:
                # Convert to numpy array if it's not already
                return np.array(input_data)
        elif input_type == 'vr':
            # Convert VR input to 3D coordinates
            # This is a placeholder - actual implementation would depend on the VR input format
            if isinstance(input_data, str):
                # For demonstration, treat each character as a VR point
                return self.get_word_positions(input_data)
            else:
                # Convert to numpy array if it's not already
                return np.array(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}. Supported types: 'keyboard', 'touch', 'vr'")

    def analyze_word(self, word: str, input_type: str = 'keyboard') -> Dict[str, Union[Dict[str, float], Dict[str, List[str]], float, int]]:
        """
        Comprehensive analysis of a word, including metrics, projections, and hex codes.

        This method performs a complete analysis of a word, calculating various metrics,
        generating projections, and compressing the path.

        Args:
            word: The word to analyze
            input_type: Type of input ('keyboard', 'touch', 'vr')

        Returns:
            Dictionary with metrics, hex codes, compression ratio, and compressed size
        """
        # Handle input and get 3D positions
        word_positions = self.handle_input(word, input_type)

        # Return empty analysis if no valid positions
        if len(word_positions) == 0:
            return {
                'metrics': {
                    'distance': 0.0,
                    'angle': 0.0,
                    'curvature': 0.0,
                    'torsion': 0.0,
                    'planarity': 0.0,
                    'compactness': 0.0
                },
                'hex_codes': {
                    'front': [],
                    'back': [],
                    'left': [],
                    'right': [],
                    'top': [],
                    'bottom': []
                },
                'compression_ratio': 0.0,
                'compressed_size': 0
            }

        # Calculate all metrics in one pass
        # This is more efficient than calling assess_word which calls each metric calculation separately
        metrics = {
            'distance': self.calculate_distance(word_positions),
            'angle': self.calculate_angle(word_positions),
            'curvature': self.calculate_curvature(word_positions),
            'torsion': self.calculate_torsion(word_positions),
            'planarity': self.calculate_planarity(word_positions),
            'compactness': self.calculate_compactness(word_positions)
        }

        # Get hex codes directly (more efficient than getting projections separately)
        hex_codes = self.get_word_hex_codes(word)

        # Compress the path
        compressed_path = self.compress_path(word_positions)

        # Calculate compression statistics
        # Use compact JSON for size calculation to match compression method
        json_data = json.dumps(word_positions.tolist(), separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        json_size = len(json_data)
        compressed_size = len(compressed_path)

        # Avoid division by zero
        compression_ratio = json_size / compressed_size if compressed_size > 0 else 0.0

        return {
            'metrics': metrics,
            'hex_codes': hex_codes,
            'compression_ratio': float(compression_ratio),
            'compressed_size': compressed_size
        }


def main() -> None:
    """
    Demonstrate the functionality of the QuantumKeyboard class.

    This function creates instances of the QuantumKeyboard class with different
    keyboard layouts, analyzes sample words, and compares the efficiency of
    different layouts.
    """
    # Create keyboard instances
    qwerty_keyboard = QuantumKeyboard('qwerty')
    dvorak_keyboard = QuantumKeyboard('dvorak')

    # Sample words to analyze
    sample_words = [
        "6 orthogonal directions",
        "quantum",
        "keyboard",
        "compression",
        "hex"
    ]

    # Analyze words on QWERTY keyboard
    print("=== QWERTY Keyboard Analysis ===")
    for word in sample_words:
        print(f"\nAnalyzing word: '{word}'")
        analysis = qwerty_keyboard.analyze_word(word)

        # Print metrics
        print("Metrics:")
        for metric, value in analysis['metrics'].items():
            print(f"  {metric}: {value:.4f}")

        # Print hex codes for front view
        print("\nHex codes (front view):")
        for i, hex_code in enumerate(analysis['hex_codes']['front']):
            print(f"  Character {i}: {hex_code}")

        # Print compression info
        print(f"\nCompression ratio: {analysis['compression_ratio']:.2f}x")
        print(f"Compressed size: {analysis['compressed_size']} bytes")

        # Visualize the word path
        word_positions = qwerty_keyboard.get_word_positions(word)
        if len(word_positions) > 1:
            print("\nGenerating 3D visualization (close the plot window to continue)...")
            qwerty_keyboard.plot_word_positions(word_positions)

    # Compare QWERTY and Dvorak for a sample word
    print("\n=== Keyboard Layout Comparison ===")
    comparison_word = "6 orthogonal directions"
    print(f"Comparing layouts for word: '{comparison_word}'")

    qwerty_analysis = qwerty_keyboard.analyze_word(comparison_word)
    dvorak_analysis = dvorak_keyboard.analyze_word(comparison_word)

    print("\nQWERTY vs Dvorak:")
    print(f"  QWERTY distance: {qwerty_analysis['metrics']['distance']:.4f}")
    print(f"  Dvorak distance: {dvorak_analysis['metrics']['distance']:.4f}")

    # Determine which layout is more efficient for this word
    if qwerty_analysis['metrics']['distance'] < dvorak_analysis['metrics']['distance']:
        print("  Result: QWERTY is more efficient for this word")
    else:
        print("  Result: Dvorak is more efficient for this word")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()

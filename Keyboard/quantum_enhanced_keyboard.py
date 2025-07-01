import numpy as np
import zlib
import gzip
import json
import io
import math
import statistics
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import seaborn as sns

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class Finger(Enum):
    LEFT_PINKY = "left_pinky"
    LEFT_RING = "left_ring"
    LEFT_MIDDLE = "left_middle"
    LEFT_INDEX = "left_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    RIGHT_INDEX = "right_index"
    RIGHT_MIDDLE = "right_middle"
    RIGHT_RING = "right_ring"
    RIGHT_PINKY = "right_pinky"

@dataclass
class QuantumKeyInfo:
    """Enhanced key information with quantum-inspired properties."""
    position: np.ndarray  # 3D position as numpy array
    finger: Finger
    hand: Hand
    effort: float = 1.0
    frequency: float = 0.0  # Letter frequency in English
    quantum_state: float = 0.0  # Quantum-inspired state value
    harmonics: List[float] = None  # Harmonic frequencies

@dataclass
class QuantumTypingStats:
    """Comprehensive quantum-enhanced typing statistics."""
    # Traditional metrics
    total_distance: float
    avg_distance_per_char: float
    hand_alternation_rate: float
    finger_utilization: Dict[Finger, int]
    same_finger_percentage: float
    bigram_efficiency: float
    trigram_efficiency: float
    
    # Quantum-inspired metrics
    curvature: float
    torsion: float
    planarity: float
    compactness: float
    quantum_coherence: float
    harmonic_resonance: float
    dimensional_complexity: float
    
    # Statistical properties
    variance: float
    entropy: float
    fractal_dimension: float

@dataclass
class StatisticalAnalysis:
    """Enhanced statistical analysis with confidence intervals."""
    mean: float
    median: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    confidence_interval_95: Tuple[float, float]
    skewness: float = 0.0
    kurtosis: float = 0.0

class QuantumKeyboard:
    """
    Enhanced quantum-inspired keyboard with comprehensive analysis capabilities.
    
    This class combines traditional typing analysis with quantum-inspired metrics,
    advanced statistical analysis, and machine learning techniques.
    """

    def __init__(self, keyboard_variant: str = 'qwerty') -> None:
        """
        Initialize a QuantumKeyboard with the specified keyboard variant.

        Args:
            keyboard_variant: The keyboard layout to use
        """
        self.keyboard_variant = keyboard_variant
        self.key_info = self.initialize_quantum_keys()
        self.bigram_cache = {}
        self.trigram_cache = {}
        self.quantum_field = self._initialize_quantum_field()
        
        # English letter frequencies for quantum state initialization
        self.letter_frequencies = {
            'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75,
            's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
            'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }

    def _initialize_quantum_field(self) -> np.ndarray:
        """Initialize a quantum field over the keyboard space."""
        # Create a 3D quantum field with harmonic oscillations
        x = np.linspace(0, 15, 50)
        y = np.linspace(0, 5, 20)
        z = np.linspace(0, 2, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Quantum field with multiple harmonics
        field = (np.sin(X * 0.5) * np.cos(Y * 0.8) * np.exp(-Z * 0.3) +
                0.5 * np.sin(X * 1.2) * np.sin(Y * 1.5) * np.cos(Z * 2.0))
        
        return field

    def initialize_quantum_keys(self) -> Dict[str, QuantumKeyInfo]:
        """
        Initialize quantum-enhanced key positions with finger assignments.

        Returns:
            Dictionary mapping characters to their quantum key information
        """
        key_info = {}
        
        if self.keyboard_variant == 'qwerty':
            # QWERTY layout with quantum enhancements
            layout_data = [
                # (char, x, y, z, finger, effort)
                ('q', 0, 1, 0, Finger.LEFT_PINKY, 1.1),
                ('w', 1, 1, 0, Finger.LEFT_RING, 1.0),
                ('e', 2, 1, 0, Finger.LEFT_MIDDLE, 0.9),
                ('r', 3, 1, 0, Finger.LEFT_INDEX, 0.9),
                ('t', 4, 1, 0, Finger.LEFT_INDEX, 1.0),
                ('y', 5, 1, 0, Finger.RIGHT_INDEX, 1.0),
                ('u', 6, 1, 0, Finger.RIGHT_INDEX, 0.9),
                ('i', 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9),
                ('o', 8, 1, 0, Finger.RIGHT_RING, 1.0),
                ('p', 9, 1, 0, Finger.RIGHT_PINKY, 1.1),
                
                # Home row
                ('a', 0, 2, 0, Finger.LEFT_PINKY, 0.8),
                ('s', 1, 2, 0, Finger.LEFT_RING, 0.7),
                ('d', 2, 2, 0, Finger.LEFT_MIDDLE, 0.6),
                ('f', 3, 2, 0, Finger.LEFT_INDEX, 0.6),
                ('g', 4, 2, 0, Finger.LEFT_INDEX, 0.7),
                ('h', 5, 2, 0, Finger.RIGHT_INDEX, 0.7),
                ('j', 6, 2, 0, Finger.RIGHT_INDEX, 0.6),
                ('k', 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6),
                ('l', 8, 2, 0, Finger.RIGHT_RING, 0.7),
                
                # Bottom row
                ('z', 0, 3, 0, Finger.LEFT_PINKY, 1.2),
                ('x', 1, 3, 0, Finger.LEFT_RING, 1.1),
                ('c', 2, 3, 0, Finger.LEFT_MIDDLE, 1.0),
                ('v', 3, 3, 0, Finger.LEFT_INDEX, 1.0),
                ('b', 4, 3, 0, Finger.LEFT_INDEX, 1.1),
                ('n', 5, 3, 0, Finger.RIGHT_INDEX, 1.1),
                ('m', 6, 3, 0, Finger.RIGHT_INDEX, 1.0),
                
                # Space
                (' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5),
            ]
            
        elif self.keyboard_variant == 'dvorak':
            # Dvorak layout
            layout_data = [
                # Top row
                ("'", 0, 1, 0, Finger.LEFT_PINKY, 1.1),
                (',', 1, 1, 0, Finger.LEFT_RING, 1.0),
                ('.', 2, 1, 0, Finger.LEFT_MIDDLE, 0.9),
                ('p', 3, 1, 0, Finger.LEFT_INDEX, 0.9),
                ('y', 4, 1, 0, Finger.LEFT_INDEX, 1.0),
                ('f', 5, 1, 0, Finger.RIGHT_INDEX, 1.0),
                ('g', 6, 1, 0, Finger.RIGHT_INDEX, 0.9),
                ('c', 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9),
                ('r', 8, 1, 0, Finger.RIGHT_RING, 1.0),
                ('l', 9, 1, 0, Finger.RIGHT_PINKY, 1.1),
                
                # Home row (optimized)
                ('a', 0, 2, 0, Finger.LEFT_PINKY, 0.8),
                ('o', 1, 2, 0, Finger.LEFT_RING, 0.7),
                ('e', 2, 2, 0, Finger.LEFT_MIDDLE, 0.6),
                ('u', 3, 2, 0, Finger.LEFT_INDEX, 0.6),
                ('i', 4, 2, 0, Finger.LEFT_INDEX, 0.7),
                ('d', 5, 2, 0, Finger.RIGHT_INDEX, 0.7),
                ('h', 6, 2, 0, Finger.RIGHT_INDEX, 0.6),
                ('t', 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6),
                ('n', 8, 2, 0, Finger.RIGHT_RING, 0.7),
                ('s', 9, 2, 0, Finger.RIGHT_PINKY, 0.8),
                
                # Bottom row
                (';', 0, 3, 0, Finger.LEFT_PINKY, 1.2),
                ('q', 1, 3, 0, Finger.LEFT_RING, 1.1),
                ('j', 2, 3, 0, Finger.LEFT_MIDDLE, 1.0),
                ('k', 3, 3, 0, Finger.LEFT_INDEX, 1.0),
                ('x', 4, 3, 0, Finger.LEFT_INDEX, 1.1),
                ('b', 5, 3, 0, Finger.RIGHT_INDEX, 1.1),
                ('m', 6, 3, 0, Finger.RIGHT_INDEX, 1.0),
                ('w', 7, 3, 0, Finger.RIGHT_MIDDLE, 1.0),
                ('v', 8, 3, 0, Finger.RIGHT_RING, 1.1),
                ('z', 9, 3, 0, Finger.RIGHT_PINKY, 1.2),
                
                # Space
                (' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5),
            ]
            
        elif self.keyboard_variant == 'colemak':
            # Colemak layout
            layout_data = [
                # Top row
                ('q', 0, 1, 0, Finger.LEFT_PINKY, 1.1),
                ('w', 1, 1, 0, Finger.LEFT_RING, 1.0),
                ('f', 2, 1, 0, Finger.LEFT_MIDDLE, 0.9),
                ('p', 3, 1, 0, Finger.LEFT_INDEX, 0.9),
                ('g', 4, 1, 0, Finger.LEFT_INDEX, 1.0),
                ('j', 5, 1, 0, Finger.RIGHT_INDEX, 1.0),
                ('l', 6, 1, 0, Finger.RIGHT_INDEX, 0.9),
                ('u', 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9),
                ('y', 8, 1, 0, Finger.RIGHT_RING, 1.0),
                (';', 9, 1, 0, Finger.RIGHT_PINKY, 1.1),
                
                # Home row
                ('a', 0, 2, 0, Finger.LEFT_PINKY, 0.8),
                ('r', 1, 2, 0, Finger.LEFT_RING, 0.7),
                ('s', 2, 2, 0, Finger.LEFT_MIDDLE, 0.6),
                ('t', 3, 2, 0, Finger.LEFT_INDEX, 0.6),
                ('d', 4, 2, 0, Finger.LEFT_INDEX, 0.7),
                ('h', 5, 2, 0, Finger.RIGHT_INDEX, 0.7),
                ('n', 6, 2, 0, Finger.RIGHT_INDEX, 0.6),
                ('e', 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6),
                ('i', 8, 2, 0, Finger.RIGHT_RING, 0.7),
                ('o', 9, 2, 0, Finger.RIGHT_PINKY, 0.8),
                
                # Bottom row
                ('z', 0, 3, 0, Finger.LEFT_PINKY, 1.2),
                ('x', 1, 3, 0, Finger.LEFT_RING, 1.1),
                ('c', 2, 3, 0, Finger.LEFT_MIDDLE, 1.0),
                ('v', 3, 3, 0, Finger.LEFT_INDEX, 1.0),
                ('b', 4, 3, 0, Finger.LEFT_INDEX, 1.1),
                ('k', 5, 3, 0, Finger.RIGHT_INDEX, 1.1),
                ('m', 6, 3, 0, Finger.RIGHT_INDEX, 1.0),
                
                # Space
                (' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5),
            ]
        else:
            # Default to empty layout
            layout_data = []

        # Create quantum key info for each character
        for char, x, y, z, finger, effort in layout_data:
            hand = Hand.LEFT if finger.value.startswith('left') else Hand.RIGHT
            freq = self.letter_frequencies.get(char, 0.0)
            
            # Calculate quantum state based on position and frequency
            quantum_state = self._calculate_quantum_state(x, y, z, freq)
            
            # Generate harmonic frequencies
            harmonics = self._generate_harmonics(x, y, z)
            
            key_info[char] = QuantumKeyInfo(
                position=np.array([x, y, z]),
                finger=finger,
                hand=hand,
                effort=effort,
                frequency=freq,
                quantum_state=quantum_state,
                harmonics=harmonics
            )

        return key_info

    def _calculate_quantum_state(self, x: float, y: float, z: float, freq: float) -> float:
        """Calculate quantum state value based on position and frequency."""
        # Combine spatial and frequency components
        spatial_component = np.sin(x * 0.3) * np.cos(y * 0.5) * np.exp(-z * 0.2)
        frequency_component = freq / 100.0  # Normalize frequency
        
        # Quantum superposition
        quantum_state = (spatial_component + frequency_component) / 2.0
        return float(quantum_state)

    def _generate_harmonics(self, x: float, y: float, z: float) -> List[float]:
        """Generate harmonic frequencies for a key position."""
        base_freq = 440.0 * (2 ** ((x + y * 12 + z * 144) / 12))  # Musical scale
        harmonics = []
        
        for i in range(1, 6):  # First 5 harmonics
            harmonic = base_freq * i
            harmonics.append(harmonic)
            
        return harmonics

    def get_word_positions(self, word: str) -> np.ndarray:
        """
        Get the 3D quantum positions for each character in a word.

        Args:
            word: The word to convert to positions

        Returns:
            A numpy array of 3D coordinates for each character in the word
        """
        positions = []
        for char in word.lower():
            if char in self.key_info:
                positions.append(self.key_info[char].position)
        
        return np.array(positions) if positions else np.array([]).reshape(0, 3)

    def calculate_quantum_coherence(self, word_positions: np.ndarray) -> float:
        """
        Calculate quantum coherence of the typing path.
        
        Coherence measures how well the path maintains quantum superposition.
        """
        if len(word_positions) < 2:
            return 0.0
            
        # Calculate phase relationships between consecutive positions
        phases = []
        for i in range(len(word_positions) - 1):
            pos1, pos2 = word_positions[i], word_positions[i + 1]
            
            # Calculate phase difference based on position vectors
            phase_diff = np.arctan2(np.cross(pos1[:2], pos2[:2]), np.dot(pos1[:2], pos2[:2]))
            phases.append(phase_diff)
        
        phases = np.array(phases)
        
        # Coherence is measured by phase stability
        if len(phases) > 1:
            coherence = 1.0 / (1.0 + np.std(phases))
        else:
            coherence = 1.0
            
        return float(coherence)

    def calculate_harmonic_resonance(self, word: str) -> float:
        """
        Calculate harmonic resonance of the word based on key harmonics.
        """
        if len(word) < 2:
            return 0.0
            
        total_resonance = 0.0
        valid_pairs = 0
        
        for i in range(len(word) - 1):
            char1, char2 = word[i].lower(), word[i + 1].lower()
            
            if char1 in self.key_info and char2 in self.key_info:
                harmonics1 = self.key_info[char1].harmonics
                harmonics2 = self.key_info[char2].harmonics
                
                # Calculate resonance between harmonic series
                resonance = 0.0
                for h1 in harmonics1[:3]:  # Use first 3 harmonics
                    for h2 in harmonics2[:3]:
                        # Resonance is stronger when frequencies are related
                        ratio = h1 / h2 if h2 != 0 else 0
                        if 0.5 <= ratio <= 2.0:  # Within an octave
                            resonance += 1.0 / abs(ratio - 1.0 + 0.1)
                
                total_resonance += resonance
                valid_pairs += 1
        
        return total_resonance / max(1, valid_pairs)

    def calculate_dimensional_complexity(self, word_positions: np.ndarray) -> float:
        """
        Calculate dimensional complexity using fractal dimension estimation.
        """
        if len(word_positions) < 3:
            return 0.0
            
        # Use box-counting method approximation
        def box_count_dimension(points, scales):
            counts = []
            for scale in scales:
                # Create grid at this scale
                grid_size = 1.0 / scale
                grid_points = set()
                
                for point in points:
                    grid_x = int(point[0] / grid_size)
                    grid_y = int(point[1] / grid_size)
                    grid_z = int(point[2] / grid_size)
                    grid_points.add((grid_x, grid_y, grid_z))
                
                counts.append(len(grid_points))
            
            # Fit line to log-log plot
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            if len(log_scales) > 1:
                slope = np.polyfit(log_scales, log_counts, 1)[0]
                return -slope  # Fractal dimension
            else:
                return 1.0
        
        # Use multiple scales
        scales = np.logspace(-1, 1, 10)
        fractal_dim = box_count_dimension(word_positions, scales)
        
        return float(np.clip(fractal_dim, 0, 3))  # Clamp to valid range

    def calculate_entropy(self, word: str) -> float:
        """
        Calculate information entropy of the word based on key positions.
        """
        if len(word) == 0:
            return 0.0
            
        # Calculate position distribution
        positions = self.get_word_positions(word)
        if len(positions) == 0:
            return 0.0
            
        # Discretize positions into bins
        bins = 10
        pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
        
        # Avoid division by zero
        ranges = pos_max - pos_min
        ranges[ranges == 0] = 1.0
        
        # Convert positions to discrete bins
        normalized_pos = (positions - pos_min) / ranges
        discrete_pos = (normalized_pos * (bins - 1)).astype(int)
        
        # Calculate entropy for each dimension
        entropies = []
        for dim in range(3):
            counts = np.bincount(discrete_pos[:, dim], minlength=bins)
            probs = counts / len(positions)
            probs = probs[probs > 0]  # Remove zero probabilities
            
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
                entropies.append(entropy)
        
        return float(np.mean(entropies)) if entropies else 0.0

    def calculate_comprehensive_quantum_stats(self, text: str) -> QuantumTypingStats:
        """
        Calculate comprehensive quantum-enhanced typing statistics.
        """
        # Get positions and basic metrics
        positions = self.get_word_positions(text)
        
        if len(positions) == 0:
            return QuantumTypingStats(
                total_distance=0.0, avg_distance_per_char=0.0, hand_alternation_rate=0.0,
                finger_utilization={}, same_finger_percentage=0.0, bigram_efficiency=0.0,
                trigram_efficiency=0.0, curvature=0.0, torsion=0.0, planarity=0.0,
                compactness=0.0, quantum_coherence=0.0, harmonic_resonance=0.0,
                dimensional_complexity=0.0, variance=0.0, entropy=0.0, fractal_dimension=0.0
            )
        
        # Traditional metrics
        total_distance = self.calculate_distance(positions)
        avg_distance_per_char = total_distance / max(1, len(text))
        hand_alternation_rate = self.calculate_hand_alternation_rate(text)
        finger_utilization = self.calculate_finger_utilization(text)
        same_finger_percentage = self.calculate_same_finger_transitions(text)[1]
        bigram_efficiency = self.calculate_bigram_efficiency(text)
        trigram_efficiency = self.calculate_trigram_efficiency(text)
        
        # Quantum-inspired metrics
        curvature = self.calculate_curvature(positions)
        torsion = self.calculate_torsion(positions)
        planarity = self.calculate_planarity(positions)
        compactness = self.calculate_compactness(positions)
        quantum_coherence = self.calculate_quantum_coherence(positions)
        harmonic_resonance = self.calculate_harmonic_resonance(text)
        dimensional_complexity = self.calculate_dimensional_complexity(positions)
        
        # Statistical metrics
        if len(positions) > 1:
            variance = float(np.var(np.linalg.norm(positions[1:] - positions[:-1], axis=1)))
        else:
            variance = 0.0
            
        entropy = self.calculate_entropy(text)
        fractal_dimension = dimensional_complexity  # Already calculated above
        
        return QuantumTypingStats(
            total_distance=total_distance,
            avg_distance_per_char=avg_distance_per_char,
            hand_alternation_rate=hand_alternation_rate,
            finger_utilization=finger_utilization,
            same_finger_percentage=same_finger_percentage,
            bigram_efficiency=bigram_efficiency,
            trigram_efficiency=trigram_efficiency,
            curvature=curvature,
            torsion=torsion,
            planarity=planarity,
            compactness=compactness,
            quantum_coherence=quantum_coherence,
            harmonic_resonance=harmonic_resonance,
            dimensional_complexity=dimensional_complexity,
            variance=variance,
            entropy=entropy,
            fractal_dimension=fractal_dimension
        )

    def calculate_hand_alternation_rate(self, text: str) -> float:
        """Calculate the percentage of hand alternations in typing."""
        hand_sequence = []
        for char in text.lower():
            if char in self.key_info:
                hand_sequence.append(self.key_info[char].hand)
        
        if len(hand_sequence) <= 1:
            return 0.0
        
        alternations = sum(1 for i in range(1, len(hand_sequence)) 
                          if hand_sequence[i] != hand_sequence[i-1])
        return (alternations / (len(hand_sequence) - 1)) * 100

    def calculate_finger_utilization(self, text: str) -> Dict[Finger, int]:
        """Calculate how many times each finger is used."""
        finger_usage = Counter()
        for char in text.lower():
            if char in self.key_info:
                finger_usage[self.key_info[char].finger] += 1
        return dict(finger_usage)

    def calculate_same_finger_transitions(self, text: str) -> Tuple[int, float]:
        """Calculate same-finger transitions (awkward typing)."""
        finger_sequence = []
        for char in text.lower():
            if char in self.key_info:
                finger_sequence.append(self.key_info[char].finger)
        
        if len(finger_sequence) <= 1:
            return 0, 0.0
        
        same_finger = sum(1 for i in range(1, len(finger_sequence)) 
                         if finger_sequence[i] == finger_sequence[i-1])
        percentage = (same_finger / (len(finger_sequence) - 1)) * 100
        return same_finger, percentage

    def calculate_bigram_efficiency(self, text: str) -> float:
        """Calculate efficiency of bigram combinations."""
        if len(text) < 2:
            return 0.0
        
        bigrams = [text[i:i+2].lower() for i in range(len(text) - 1)]
        total_efficiency = 0.0
        valid_bigrams = 0
        
        for bigram in bigrams:
            if len(bigram) == 2 and all(c in self.key_info for c in bigram):
                char1, char2 = bigram
                info1, info2 = self.key_info[char1], self.key_info[char2]
                
                efficiency = 1.0
                
                # Hand alternation bonus
                if info1.hand != info2.hand:
                    efficiency *= 1.2
                
                # Same finger penalty
                if info1.finger == info2.finger:
                    efficiency *= 0.3
                
                # Distance penalty
                distance = np.linalg.norm(info1.position - info2.position)
                efficiency *= max(0.1, 1.0 - (distance / 10.0))
                
                # Effort consideration
                efficiency *= (2.0 - info1.effort) * (2.0 - info2.effort) / 4.0
                
                total_efficiency += efficiency
                valid_bigrams += 1
        
        return total_efficiency / max(1, valid_bigrams)

    def calculate_trigram_efficiency(self, text: str) -> float:
        """Calculate efficiency of trigram combinations."""
        if len(text) < 3:
            return 0.0
        
        trigrams = [text[i:i+3].lower() for i in range(len(text) - 2)]
        total_efficiency = 0.0
        valid_trigrams = 0
        
        for trigram in trigrams:
            if len(trigram) == 3 and all(c in self.key_info for c in trigram):
                chars = [self.key_info[c] for c in trigram]
                
                efficiency = 1.0
                
                # Hand alternation pattern
                hands = [info.hand for info in chars]
                if hands[0] != hands[1] and hands[1] != hands[2]:
                    efficiency *= 1.3
                elif hands[0] == hands[1] == hands[2]:
                    efficiency *= 0.7
                
                # Same finger penalty
                fingers = [info.finger for info in chars]
                unique_fingers = len(set(fingers))
                if unique_fingers == 1:
                    efficiency *= 0.2
                elif unique_fingers == 2:
                    efficiency *= 0.6
                
                # Path distance
                dist1 = np.linalg.norm(chars[0].position - chars[1].position)
                dist2 = np.linalg.norm(chars[1].position - chars[2].position)
                total_distance = dist1 + dist2
                efficiency *= max(0.1, 1.0 - (total_distance / 15.0))
                
                total_efficiency += efficiency
                valid_trigrams += 1
        
        return total_efficiency / max(1, valid_trigrams)

    def calculate_distance(self, word_positions: np.ndarray) -> float:
        """Calculate the total distance of the path formed by the word."""
        if len(word_positions) < 2:
            return 0.0

        diffs = word_positions[1:] - word_positions[:-1]
        distance = np.sum(np.linalg.norm(diffs, axis=1))
        return float(distance)

    def calculate_angle(self, word_positions: np.ndarray) -> float:
        """Calculate the average angle between consecutive segments."""
        if len(word_positions) < 3:
            return 0.0

        vectors = word_positions[1:] - word_positions[:-1]
        vector_norms = np.linalg.norm(vectors, axis=1)

        valid_indices = np.where(vector_norms > 0)[0]
        if len(valid_indices) < 2:
            return 0.0

        vectors = vectors[valid_indices]
        vector_norms = vector_norms[valid_indices]
        normalized_vectors = vectors / vector_norms[:, np.newaxis]

        dot_products = np.sum(normalized_vectors[:-1] * normalized_vectors[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)

        return float(np.mean(angles)) if len(angles) > 0 else 0.0

    def calculate_curvature(self, word_positions: np.ndarray) -> float:
        """Calculate the average curvature of the word path."""
        if len(word_positions) < 3:
            return 0.0

        vectors = word_positions[1:] - word_positions[:-1]
        vector_norms = np.linalg.norm(vectors, axis=1)

        valid_indices = np.where(vector_norms > 0)[0]
        if len(valid_indices) < 2:
            return 0.0

        vectors = vectors[valid_indices]
        vector_norms = vector_norms[valid_indices]

        cross_products = np.cross(vectors[:-1], vectors[1:])
        cross_norms = np.linalg.norm(cross_products, axis=1)
        norm_products = vector_norms[:-1] * vector_norms[1:]

        valid_indices = norm_products > 0
        if not np.any(valid_indices):
            return 0.0

        curvatures = cross_norms[valid_indices] / norm_products[valid_indices]
        return float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0

    def calculate_torsion(self, word_positions: np.ndarray) -> float:
        """Calculate the average torsion of the word path."""
        if len(word_positions) < 4:
            return 0.0

        v1 = word_positions[1:-2] - word_positions[0:-3]
        v2 = word_positions[2:-1] - word_positions[1:-2]
        v3 = word_positions[3:] - word_positions[2:-1]

        cross1 = np.cross(v1, v2)
        cross2 = np.cross(v2, v3)

        cross1_norms = np.linalg.norm(cross1, axis=1)
        cross2_norms = np.linalg.norm(cross2, axis=1)

        valid_indices = (cross1_norms > 0) & (cross2_norms > 0)
        if not np.any(valid_indices):
            return 0.0

        valid_cross1 = cross1[valid_indices]
        valid_cross2 = cross2[valid_indices]
        valid_cross1_norms = cross1_norms[valid_indices]
        valid_cross2_norms = cross2_norms[valid_indices]

        dot_products = np.sum(valid_cross1 * valid_cross2, axis=1)
        cos_torsions = np.clip(dot_products / (valid_cross1_norms * valid_cross2_norms), -1.0, 1.0)

        return float(np.mean(cos_torsions)) if len(cos_torsions) > 0 else 0.0

    def calculate_planarity(self, word_positions: np.ndarray) -> float:
        """Calculate the planarity using PCA."""
        if len(word_positions) < 3:
            return 0.0

        pca = PCA(n_components=2)
        pca.fit(word_positions)
        planarity = sum(pca.explained_variance_ratio_)
        return float(planarity)

    def calculate_compactness(self, word_positions: np.ndarray) -> float:
        """Calculate the compactness of the word path."""
        if len(word_positions) < 2:
            return 0.0

        diffs = word_positions[1:] - word_positions[:-1]
        consecutive_distances = np.linalg.norm(diffs, axis=1)
        avg_consecutive_distance = np.mean(consecutive_distances) if len(consecutive_distances) > 0 else 0.0

        # Calculate maximum distance
        max_distance = 0.0
        if len(word_positions) >= 2:
            try:
                from scipy.spatial.distance import pdist
                max_distance = np.max(pdist(word_positions))
            except ImportError:
                max_x = np.ptp(word_positions[:, 0])
                max_y = np.ptp(word_positions[:, 1])
                max_z = np.ptp(word_positions[:, 2])
                max_distance = np.sqrt(max_x**2 + max_y**2 + max_z**2)

        if max_distance == 0:
            return 0.0

        compactness = avg_consecutive_distance / max_distance
        return float(compactness)

    def plot_quantum_analysis(self, word: str, show_field: bool = True) -> None:
        """
        Create comprehensive quantum visualization of word analysis.
        """
        positions = self.get_word_positions(word)
        if len(positions) == 0:
            print("No positions to plot")
            return

        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D Path with Quantum Field
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        if show_field and hasattr(self, 'quantum_field'):
            # Sample quantum field for visualization
            field_sample = self.quantum_field[::5, ::2, ::1]  # Downsample for performance
            x_field = np.linspace(0, 15, field_sample.shape[0])
            y_field = np.linspace(0, 5, field_sample.shape[1])
            z_field = np.linspace(0, 2, field_sample.shape[2])
            
            # Plot field as scattered points with transparency
            X_field, Y_field, Z_field = np.meshgrid(x_field, y_field, z_field, indexing='ij')
            field_flat = field_sample.flatten()
            
            # Only show high-intensity field points
            high_intensity = field_flat > np.percentile(field_flat, 90)
            if np.any(high_intensity):
                ax1.scatter(X_field.flatten()[high_intensity], 
                           Y_field.flatten()[high_intensity], 
                           Z_field.flatten()[high_intensity],
                           c=field_flat[high_intensity], cmap='plasma', 
                           alpha=0.1, s=1)

        # Plot word path
        x_coords, y_coords, z_coords = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Color points by quantum states
        colors = []
        for char in word.lower():
            if char in self.key_info:
                colors.append(self.key_info[char].quantum_state)
        
        scatter = ax1.scatter(x_coords, y_coords, z_coords, 
                             c=colors, cmap='viridis', s=100, alpha=0.8)
        ax1.plot3D(x_coords, y_coords, z_coords, 'b-', linewidth=3, alpha=0.7)
        
        # Add character labels
        for i, (pos, char) in enumerate(zip(positions, word.lower())):
            ax1.text(pos[0], pos[1], pos[2], char.upper(), 
                    fontsize=12, ha='center', va='center', weight='bold')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Quantum Path: "{word}"', fontsize=14)
        plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Quantum State')

        # 2. Quantum Metrics Radar Chart
        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        
        stats = self.calculate_comprehensive_quantum_stats(word)
        metrics = {
            'Distance': stats.total_distance / 10,  # Normalize
            'Curvature': stats.curvature,
            'Planarity': stats.planarity,
            'Quantum Coherence': stats.quantum_coherence,
            'Harmonic Resonance': min(stats.harmonic_resonance / 10, 1.0),
            'Dimensional Complexity': stats.dimensional_complexity / 3,
            'Entropy': stats.entropy / 5,
            'Hand Alternation': stats.hand_alternation_rate / 100
        }
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = list(metrics.values())
        
        ax2.plot(angles, values, 'o-', linewidth=2)
        ax2.fill(angles, values, alpha=0.25)
        ax2.set_xticks(angles)
        ax2.set_xticklabels(metrics.keys())
        ax2.set_ylim(0, 1)
        ax2.set_title('Quantum Metrics Profile', fontsize=14)

        # 3. Harmonic Frequency Spectrum
        ax3 = fig.add_subplot(2, 3, 3)
        
        all_harmonics = []
        harmonic_labels = []
        for char in word.lower():
            if char in self.key_info:
                harmonics = self.key_info[char].harmonics[:3]  # First 3 harmonics
                all_harmonics.extend(harmonics)
                harmonic_labels.extend([f'{char.upper()}_{i+1}' for i in range(len(harmonics))])
        
        if all_harmonics:
            bars = ax3.bar(range(len(all_harmonics)), all_harmonics, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(all_harmonics))))
            ax3.set_xlabel('Harmonic Index')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_title('Harmonic Spectrum')
            ax3.set_xticks(range(len(all_harmonics)))
            ax3.set_xticklabels(harmonic_labels, rotation=45, ha='right')

        # 4. Dimensional Reduction (t-SNE or PCA)
        ax4 = fig.add_subplot(2, 3, 4)
        
        if len(positions) >= 3:
            # Use PCA for 2D projection
            pca = PCA(n_components=2)
            projected = pca.fit_transform(positions)
            
            scatter = ax4.scatter(projected[:, 0], projected[:, 1], 
                                 c=colors, cmap='viridis', s=100)
            
            # Connect points in order
            ax4.plot(projected[:, 0], projected[:, 1], 'b-', alpha=0.5)
            
            # Add labels
            for i, char in enumerate(word.lower()):
                ax4.annotate(char.upper(), (projected[i, 0], projected[i, 1]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax4.set_title('PCA Projection')

        # 5. Finger Usage Distribution
        ax5 = fig.add_subplot(2, 3, 5)
        
        finger_usage = stats.finger_utilization
        if finger_usage:
            fingers = list(finger_usage.keys())
            counts = list(finger_usage.values())
            
            bars = ax5.bar([f.value.replace('_', '\n') for f in fingers], counts,
                          color=plt.cm.Set3(np.linspace(0, 1, len(fingers))))
            ax5.set_xlabel('Finger')
            ax5.set_ylabel('Usage Count')
            ax5.set_title('Finger Utilization')
            ax5.tick_params(axis='x', rotation=45)

        # 6. Quantum State Evolution
        ax6 = fig.add_subplot(2, 3, 6)
        
        quantum_states = []
        char_positions = []
        for i, char in enumerate(word.lower()):
            if char in self.key_info:
                quantum_states.append(self.key_info[char].quantum_state)
                char_positions.append(i)
        
        if quantum_states:
            ax6.plot(char_positions, quantum_states, 'o-', linewidth=2, markersize=8)
            ax6.fill_between(char_positions, quantum_states, alpha=0.3)
            ax6.set_xlabel('Character Position')
            ax6.set_ylabel('Quantum State')
            ax6.set_title('Quantum State Evolution')
            ax6.grid(True, alpha=0.3)
            
            # Add character labels
            for pos, state, char in zip(char_positions, quantum_states, word.lower()):
                ax6.annotate(char.upper(), (pos, state), 
                           xytext=(0, 10), textcoords='offset points', 
                           ha='center')

        plt.tight_layout()
        plt.show()

    def analyze_multiple_layouts(self, texts: List[str], layouts: List[str]) -> Dict[str, Dict[str, StatisticalAnalysis]]:
        """
        Analyze multiple texts across different keyboard layouts with statistical analysis.
        """
        results = {}
        
        for layout_name in layouts:
            keyboard = QuantumKeyboard(layout_name)
            layout_metrics = {
                'total_distance': [],
                'quantum_coherence': [],
                'harmonic_resonance': [],
                'dimensional_complexity': [],
                'entropy': [],
                'hand_alternation_rate': [],
                'bigram_efficiency': [],
                'curvature': [],
                'planarity': []
            }
            
            for text in texts:
                stats = keyboard.calculate_comprehensive_quantum_stats(text)
                layout_metrics['total_distance'].append(stats.total_distance)
                layout_metrics['quantum_coherence'].append(stats.quantum_coherence)
                layout_metrics['harmonic_resonance'].append(stats.harmonic_resonance)
                layout_metrics['dimensional_complexity'].append(stats.dimensional_complexity)
                layout_metrics['entropy'].append(stats.entropy)
                layout_metrics['hand_alternation_rate'].append(stats.hand_alternation_rate)
                layout_metrics['bigram_efficiency'].append(stats.bigram_efficiency)
                layout_metrics['curvature'].append(stats.curvature)
                layout_metrics['planarity'].append(stats.planarity)
            
            # Calculate statistical analysis for each metric
            results[layout_name] = {}
            for metric_name, values in layout_metrics.items():
                if values:
                    mean_val = statistics.mean(values)
                    median_val = statistics.median(values)
                    
                    if len(values) > 1:
                        std_dev = statistics.stdev(values)
                        variance = statistics.variance(values)
                    else:
                        std_dev = 0.0
                        variance = 0.0
                    
                    min_val = min(values)
                    max_val = max(values)
                    
                    # 95% confidence interval
                    if len(values) > 1:
                        margin_error = 1.96 * (std_dev / math.sqrt(len(values)))
                        ci_lower = mean_val - margin_error
                        ci_upper = mean_val + margin_error
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    # Calculate skewness and kurtosis if possible
                    try:
                        import scipy.stats as stats_scipy
                        skewness = float(stats_scipy.skew(values))
                        kurtosis = float(stats_scipy.kurtosis(values))
                    except ImportError:
                        skewness = 0.0
                        kurtosis = 0.0
                    
                    results[layout_name][metric_name] = StatisticalAnalysis(
                        mean=mean_val,
                        median=median_val,
                        std_dev=std_dev,
                        variance=variance,
                        min_value=min_val,
                        max_value=max_val,
                        confidence_interval_95=(ci_lower, ci_upper),
                        skewness=skewness,
                        kurtosis=kurtosis
                    )
        
        return results

    def create_comparative_visualization(self, texts: List[str], layouts: List[str]) -> None:
        """
        Create comprehensive comparative visualization across layouts.
        """
        # Analyze all layouts
        results = self.analyze_multiple_layouts(texts, layouts)
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        metrics_to_plot = [
            'total_distance', 'quantum_coherence', 'harmonic_resonance',
            'dimensional_complexity', 'entropy', 'hand_alternation_rate',
            'bigram_efficiency', 'curvature', 'planarity'
        ]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(layouts)))
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            means = []
            stds = []
            layout_names = []
            
            for layout in layouts:
                if metric in results[layout]:
                    stat = results[layout][metric]
                    means.append(stat.mean)
                    stds.append(stat.std_dev)
                    layout_names.append(layout)
            
            if means:
                bars = ax.bar(layout_names, means, yerr=stds, 
                             color=colors[:len(layout_names)], 
                             alpha=0.7, capsize=5)
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle('Quantum Keyboard Layout Comparison', fontsize=16, y=1.02)
        plt.show()

    def assess_word(self, word: str) -> Dict[str, float]:
        """
        Calculate quantum-enhanced metrics for a word.
        Maintains compatibility with original interface while adding quantum features.
        """
        stats = self.calculate_comprehensive_quantum_stats(word)
        
        return {
            'distance': stats.total_distance,
            'angle': self.calculate_angle(self.get_word_positions(word)),
            'curvature': stats.curvature,
            'torsion': stats.torsion,
            'planarity': stats.planarity,
            'compactness': stats.compactness,
            'quantum_coherence': stats.quantum_coherence,
            'harmonic_resonance': stats.harmonic_resonance,
            'dimensional_complexity': stats.dimensional_complexity,
            'entropy': stats.entropy
        }

    def project_orthogonal_3d_path(self, word_positions: np.ndarray) -> Dict[str, np.ndarray]:
        """Project a 3D path onto 6 orthogonal directions."""
        if len(word_positions) == 0:
            empty_array = np.array([]).reshape(0, 2)
            return {
                "front": empty_array, "back": empty_array,
                "left": empty_array, "right": empty_array,
                "top": empty_array, "bottom": empty_array
            }

        if len(word_positions.shape) == 2 and word_positions.shape[1] >= 3:
            x, y, z = word_positions[:, 0], word_positions[:, 1], word_positions[:, 2]

            projections = {
                "front": np.column_stack((x, y)),
                "back": np.column_stack((x, -y)),
                "left": np.column_stack((y, z)),
                "right": np.column_stack((-y, z)),
                "top": np.column_stack((x, z)),
                "bottom": np.column_stack((x, -z))
            }
            return projections
        else:
            empty_array = np.array([]).reshape(0, 2)
            return {
                "front": empty_array, "back": empty_array,
                "left": empty_array, "right": empty_array,
                "top": empty_array, "bottom": empty_array
            }

    def to_hex(self, projection: np.ndarray) -> List[str]:
        """Convert 2D coordinates to hex code."""
        if len(projection) == 0:
            return []

        if not isinstance(projection, np.ndarray):
            projection = np.array(projection)

        clamped = np.clip(projection.astype(int), 0, 255)
        hex_codes = []
        for point in clamped:
            hex_digits = ''.join(hex(val)[2:].zfill(2).upper() for val in point)
            hex_codes.append(hex_digits)

        return hex_codes

    def compress_path(self, path: np.ndarray) -> bytes:
        """Compress a 3D path using zlib compression."""
        if len(path) == 0:
            return b''

        path_list = path.tolist() if isinstance(path, np.ndarray) else path
        json_data = json.dumps(path_list, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return zlib.compress(json_data, level=9)

    def decompress_path(self, compressed_path: bytes) -> np.ndarray:
        """Decompress a 3D path."""
        if not compressed_path:
            return np.array([]).reshape(0, 3)

        try:
            decompressed_data = zlib.decompress(compressed_path).decode('utf-8')
            path_list = json.loads(decompressed_data)
            return np.array(path_list)
        except Exception as e:
            print(f"Error decompressing path: {e}")
            return np.array([]).reshape(0, 3)

    def gzip_compress_path(self, path: np.ndarray) -> bytes:
        """Compress using gzip."""
        if len(path) == 0:
            return b''

        path_list = path.tolist() if isinstance(path, np.ndarray) else path
        json_data = json.dumps(path_list, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=9) as f:
            f.write(json_data)
        return buffer.getvalue()

    def gzip_decompress_path(self, compressed_path: bytes) -> np.ndarray:
        """Decompress using gzip."""
        if not compressed_path:
            return np.array([]).reshape(0, 3)

        try:
            buffer = io.BytesIO(compressed_path)
            with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
                decompressed_data = f.read().decode('utf-8')
            path_list = json.loads(decompressed_data)
            return np.array(path_list)
        except Exception as e:
            print(f"Error decompressing gzip path: {e}")
            return np.array([]).reshape(0, 3)

    def get_word_hex_codes(self, word: str) -> Dict[str, List[str]]:
        """Get hex codes for a word in 6 orthogonal directions."""
        word_positions = self.get_word_positions(word)
        
        if len(word_positions) == 0:
            return {direction: [] for direction in ["front", "back", "left", "right", "top", "bottom"]}

        projections = self.project_orthogonal_3d_path(word_positions)
        hex_codes = {direction: self.to_hex(projection) 
                    for direction, projection in projections.items()}

        return hex_codes

    def analyze_word(self, word: str, input_type: str = 'keyboard') -> Dict[str, Union[Dict[str, float], Dict[str, List[str]], float, int]]:
        """
        Comprehensive quantum analysis of a word.
        """
        word_positions = self.get_word_positions(word)

        if len(word_positions) == 0:
            return {
                'metrics': {},
                'hex_codes': {direction: [] for direction in ["front", "back", "left", "right", "top", "bottom"]},
                'compression_ratio': 0.0,
                'compressed_size': 0,
                'quantum_stats': None
            }

        # Get comprehensive metrics
        metrics = self.assess_word(word)
        quantum_stats = self.calculate_comprehensive_quantum_stats(word)
        hex_codes = self.get_word_hex_codes(word)

        # Compression analysis
        compressed_path = self.compress_path(word_positions)
        json_data = json.dumps(word_positions.tolist(), separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        json_size = len(json_data)
        compressed_size = len(compressed_path)
        compression_ratio = json_size / compressed_size if compressed_size > 0 else 0.0

        return {
            'metrics': metrics,
            'hex_codes': hex_codes,
            'compression_ratio': float(compression_ratio),
            'compressed_size': compressed_size,
            'quantum_stats': quantum_stats
        }


def main() -> None:
    """
    Demonstrate the enhanced QuantumKeyboard functionality.
    """
    print("=== QUANTUM KEYBOARD ANALYZER ===\n")
    
    # Create quantum keyboards for different layouts
    layouts = ['qwerty', 'dvorak', 'colemak']
    keyboards = {layout: QuantumKeyboard(layout) for layout in layouts}
    
    # Enhanced sample texts for analysis
    sample_texts = [
        "quantum",
        "keyboard", 
        "analysis",
        "6 orthogonal directions",
        "harmonic resonance patterns",
        "dimensional complexity measurement",
        "quantum coherence in typing",
        "statistical significance testing"
    ]
    
    # Single word quantum analysis demonstration
    demo_word = "quantum"
    print(f"=== QUANTUM ANALYSIS: '{demo_word}' ===")
    
    qwerty_kb = keyboards['qwerty']
    analysis = qwerty_kb.analyze_word(demo_word)
    
    print("Enhanced Metrics:")
    for metric, value in analysis['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nCompression: {analysis['compression_ratio']:.2f}x")
    print(f"Compressed size: {analysis['compressed_size']} bytes")
    
    # Show quantum stats
    if analysis['quantum_stats']:
        stats = analysis['quantum_stats']
        print(f"\nQuantum Properties:")
        print(f"  Quantum Coherence: {stats.quantum_coherence:.4f}")
        print(f"  Harmonic Resonance: {stats.harmonic_resonance:.4f}")
        print(f"  Dimensional Complexity: {stats.dimensional_complexity:.4f}")
        print(f"  Information Entropy: {stats.entropy:.4f}")
    
    # Multi-layout statistical comparison
    print(f"\n=== STATISTICAL LAYOUT COMPARISON ===")
    
    statistical_results = qwerty_kb.analyze_multiple_layouts(sample_texts, layouts)
    
    # Show key metrics comparison
    key_metrics = ['total_distance', 'quantum_coherence', 'harmonic_resonance', 'entropy']
    
    print(f"\n{'Metric':<20} | {'Layout':<10} | {'Mean':<8} | {'Std Dev':<8} | {'95% CI':<20}")
    print("-" * 80)
    
    for metric in key_metrics:
        for layout in layouts:
            if metric in statistical_results[layout]:
                stat = statistical_results[layout][metric]
                ci_str = f"({stat.confidence_interval_95[0]:.3f}, {stat.confidence_interval_95[1]:.3f})"
                print(f"{metric:<20} | {layout:<10} | {stat.mean:<8.3f} | {stat.std_dev:<8.3f} | {ci_str:<20}")
    
    # Determine optimal layout
    print(f"\n=== QUANTUM LAYOUT RECOMMENDATION ===")
    
    # Calculate composite scores for each layout
    layout_scores = {}
    weights = {
        'total_distance': -1.0,  # Lower is better
        'quantum_coherence': 2.0,  # Higher is better
        'harmonic_resonance': 1.5,  # Higher is better
        'entropy': 1.0,  # Higher is better
        'hand_alternation_rate': 1.0,  # Higher is better
        'bigram_efficiency': 1.5,  # Higher is better
        'dimensional_complexity': 0.5  # Moderate complexity is good
    }
    
    for layout in layouts:
        score = 0.0
        for metric, weight in weights.items():
            if metric in statistical_results[layout]:
                metric_value = statistical_results[layout][metric].mean
                score += weight * metric_value
        layout_scores[layout] = score
    
    optimal_layout = max(layout_scores.items(), key=lambda x: x[1])
    print(f"Recommended Layout: {optimal_layout[0].upper()}")
    print(f"Quantum Score: {optimal_layout[1]:.3f}")
    
    print(f"\nAll Layout Scores:")
    for layout, score in sorted(layout_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layout.upper()}: {score:.3f}")
    
    # Visualization demonstration
    print(f"\n=== VISUALIZATION DEMO ===")
    print("Generating quantum analysis visualization...")
    
    try:
        # Create quantum visualization for the demo word
        qwerty_kb.plot_quantum_analysis(demo_word, show_field=True)
        
        # Create comparative visualization
        print("Generating comparative layout analysis...")
        qwerty_kb.create_comparative_visualization(sample_texts[:5], layouts)
        
    except Exception as e:
        print(f"Visualization requires matplotlib. Error: {e}")
        print("Install matplotlib to see visualizations: pip install matplotlib seaborn")
    
    # Advanced analysis features demonstration
    print(f"\n=== ADVANCED FEATURES DEMO ===")
    
    # Hex code generation
    hex_codes = qwerty_kb.get_word_hex_codes(demo_word)
    print(f"\nHex codes for '{demo_word}' (first 3 characters):")
    for direction, codes in hex_codes.items():
        if codes:
            print(f"  {direction}: {' '.join(codes[:3])}")
    
    # Compression comparison
    print(f"\nCompression Analysis for '{demo_word}':")
    positions = qwerty_kb.get_word_positions(demo_word)
    
    original_size = len(json.dumps(positions.tolist()).encode('utf-8'))
    zlib_compressed = qwerty_kb.compress_path(positions)
    gzip_compressed = qwerty_kb.gzip_compress_path(positions)
    
    print(f"  Original: {original_size} bytes")
    print(f"  Zlib: {len(zlib_compressed)} bytes ({len(zlib_compressed)/original_size*100:.1f}%)")
    print(f"  Gzip: {len(gzip_compressed)} bytes ({len(gzip_compressed)/original_size*100:.1f}%)")
    
    # Verify decompression
    decompressed_zlib = qwerty_kb.decompress_path(zlib_compressed)
    decompressed_gzip = qwerty_kb.gzip_decompress_path(gzip_compressed)
    
    print(f"  Decompression verification:")
    print(f"    Zlib: {'✓' if np.allclose(positions, decompressed_zlib) else '✗'}")
    print(f"    Gzip: {'✓' if np.allclose(positions, decompressed_gzip) else '✗'}")
    
    # Machine learning features demo
    print(f"\n=== MACHINE LEARNING FEATURES ===")
    
    try:
        # Demonstrate dimensionality reduction
        all_words = ["quantum", "keyboard", "analysis", "typing", "efficiency"]
        all_positions = []
        labels = []
        
        for word in all_words:
            pos = qwerty_kb.get_word_positions(word)
            if len(pos) > 0:
                # Flatten positions for ML analysis
                flat_pos = pos.flatten()
                # Pad or truncate to fixed length
                fixed_length = 30  # 10 characters * 3 dimensions
                if len(flat_pos) < fixed_length:
                    flat_pos = np.pad(flat_pos, (0, fixed_length - len(flat_pos)))
                else:
                    flat_pos = flat_pos[:fixed_length]
                
                all_positions.append(flat_pos)
                labels.append(word)
        
        if len(all_positions) > 2:
            all_positions = np.array(all_positions)
            
            # PCA analysis
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(all_positions)
            
            print(f"PCA Analysis of word patterns:")
            print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"  Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
            
            # Try clustering if we have enough samples
            if len(all_positions) >= 3:
                try:
                    kmeans = KMeans(n_clusters=min(3, len(all_positions)), random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(all_positions)
                    
                    print(f"\nK-means clustering results:")
                    for word, cluster in zip(labels, clusters):
                        print(f"  {word}: Cluster {cluster}")
                        
                except Exception as e:
                    print(f"Clustering analysis requires more samples: {e}")
    
    except ImportError:
        print("Advanced ML features require scikit-learn: pip install scikit-learn")
    
    # Performance benchmarking
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    
    import time
    
    # Benchmark different operations
    benchmark_text = "the quick brown fox jumps over the lazy dog"
    iterations = 100
    
    # Time quantum analysis
    start_time = time.time()
    for _ in range(iterations):
        qwerty_kb.calculate_comprehensive_quantum_stats(benchmark_text)
    quantum_time = (time.time() - start_time) / iterations
    
    # Time traditional analysis
    start_time = time.time()
    for _ in range(iterations):
        qwerty_kb.assess_word(benchmark_text)
    traditional_time = (time.time() - start_time) / iterations
    
    # Time path compression
    positions = qwerty_kb.get_word_positions(benchmark_text)
    start_time = time.time()
    for _ in range(iterations):
        compressed = qwerty_kb.compress_path(positions)
        qwerty_kb.decompress_path(compressed)
    compression_time = (time.time() - start_time) / iterations
    
    print(f"Performance Benchmarks (avg over {iterations} iterations):")
    print(f"  Quantum Analysis: {quantum_time*1000:.3f} ms")
    print(f"  Traditional Analysis: {traditional_time*1000:.3f} ms")
    print(f"  Compression Round-trip: {compression_time*1000:.3f} ms")
    
    # Show efficiency gain
    if traditional_time > 0:
        overhead = ((quantum_time - traditional_time) / traditional_time) * 100
        print(f"  Quantum Analysis Overhead: {overhead:+.1f}%")
    
    # Final summary
    print(f"\n=== SUMMARY ===")
    print(f"Analyzed {len(sample_texts)} texts across {len(layouts)} keyboard layouts")
    print(f"Generated {len(key_metrics)} statistical metrics with confidence intervals")
    print(f"Optimal layout recommendation: {optimal_layout[0].upper()}")
    print(f"Quantum enhancement provides {len(analysis['metrics']) - 6} additional metrics")
    print(f"Compression achieved up to {max(len(zlib_compressed)/original_size, len(gzip_compressed)/original_size)*100:.1f}% size reduction")
    
    print(f"\n✨ Quantum Keyboard Analysis Complete! ✨")


# Additional utility functions for advanced analysis
def compare_quantum_keyboards(words: List[str], layouts: List[str] = None) -> Dict:
    """
    Comprehensive comparison function for batch analysis.
    """
    if layouts is None:
        layouts = ['qwerty', 'dvorak', 'colemak']
    
    keyboards = {layout: QuantumKeyboard(layout) for layout in layouts}
    results = {}
    
    for layout_name, keyboard in keyboards.items():
        layout_results = []
        
        for word in words:
            analysis = keyboard.analyze_word(word)
            quantum_stats = keyboard.calculate_comprehensive_quantum_stats(word)
            
            result = {
                'word': word,
                'metrics': analysis['metrics'],
                'quantum_coherence': quantum_stats.quantum_coherence,
                'harmonic_resonance': quantum_stats.harmonic_resonance,
                'dimensional_complexity': quantum_stats.dimensional_complexity,
                'entropy': quantum_stats.entropy,
                'compression_ratio': analysis['compression_ratio']
            }
            
            layout_results.append(result)
        
        results[layout_name] = layout_results
    
    return results

def export_analysis_results(results: Dict, filename: str = "quantum_keyboard_analysis.json") -> None:
    """
    Export analysis results to JSON file.
    """
    # Convert numpy arrays and complex objects to JSON-serializable format
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    try:
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"Results exported to {filename}")
    except Exception as e:
        print(f"Error exporting results: {e}")

def load_analysis_results(filename: str = "quantum_keyboard_analysis.json") -> Dict:
    """
    Export analysis results to JSON file.
    """
    # Convert numpy arrays and complex objects to JSON-serializable format
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    try:
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"Results exported to {filename}")
    except Exception as e:
        print(f"Error exporting results: {e}")

def load_analysis_results(filename: str = "quantum_keyboard_analysis.json") -> Dict:
    """
    Load analysis results from JSON file.
    """
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"Results loaded from {filename}")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}


if __name__ == "__main__":
    main()
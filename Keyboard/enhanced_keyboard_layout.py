import zlib
import gzip
import json
import math
import io
import statistics
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

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
class KeyInfo:
    """Information about a key including position, finger assignment, and hand."""
    position: Tuple[float, float, float]
    finger: Finger
    hand: Hand
    effort: float = 1.0  # Relative effort to press this key (1.0 = normal)

@dataclass
class TypingStats:
    """Comprehensive typing statistics."""
    total_distance: float
    avg_distance_per_char: float
    same_row_transitions: int
    same_row_percentage: float
    hand_alternation_rate: float
    finger_utilization: Dict[Finger, int]
    most_used_fingers: List[Tuple[Finger, int]]
    total_effort: float
    avg_effort_per_char: float
    same_finger_transitions: int
    same_finger_percentage: float
    bigram_efficiency: float
    trigram_efficiency: float

@dataclass
class StatisticalAnalysis:
    """Statistical analysis of layout performance."""
    mean: float
    median: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    confidence_interval_95: Tuple[float, float]

class KeyboardLayout(ABC):
    """Enhanced base class for keyboard layouts with comprehensive analysis."""

    def __init__(self, layout_name: str):
        """Initialize a keyboard layout with a name."""
        self.layout_name = layout_name
        self.key_info: Dict[str, KeyInfo] = {}
        self.bigram_cache: Dict[str, float] = {}
        self.trigram_cache: Dict[str, float] = {}

    @abstractmethod
    def set_key_positions(self) -> None:
        """Set key positions for the keyboard layout."""
        pass

    def set_key_info(self, char: str, x: float, y: float, z: float = 0, 
                     finger: Finger = Finger.RIGHT_INDEX, effort: float = 1.0) -> None:
        """Assign comprehensive information to a key."""
        hand = Hand.LEFT if finger.value.startswith('left') else Hand.RIGHT
        self.key_info[char.lower()] = KeyInfo(
            position=(x, y, z),
            finger=finger,
            hand=hand,
            effort=effort
        )

    def get_key_info(self, char: str) -> Optional[KeyInfo]:
        """Get comprehensive key information."""
        return self.key_info.get(char.lower())

    def get_key_position(self, char: str) -> Optional[Tuple[float, float, float]]:
        """Get the 3D coordinates for a key."""
        info = self.get_key_info(char)
        return info.position if info else None

    def get_3d_path(self, word: str) -> List[Tuple[float, float, float]]:
        """Return a list of 3D coordinates for each character in the word."""
        path = []
        for char in word.lower():
            pos = self.get_key_position(char)
            if pos:
                path.append(pos)
        return path

    def get_finger_sequence(self, text: str) -> List[Finger]:
        """Get the sequence of fingers used to type text."""
        sequence = []
        for char in text.lower():
            info = self.get_key_info(char)
            if info:
                sequence.append(info.finger)
        return sequence

    def get_hand_sequence(self, text: str) -> List[Hand]:
        """Get the sequence of hands used to type text."""
        sequence = []
        for char in text.lower():
            info = self.get_key_info(char)
            if info:
                sequence.append(info.hand)
        return sequence

class QWERTYLayout(KeyboardLayout):
    """Enhanced QWERTY keyboard layout with finger assignments."""

    def __init__(self):
        super().__init__("QWERTY")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0) with finger assignments and effort values
        self.set_key_info("`", 0, 0, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("1", 1, 0, 0, Finger.LEFT_PINKY, 1.3)
        self.set_key_info("2", 2, 0, 0, Finger.LEFT_RING, 1.2)
        self.set_key_info("3", 3, 0, 0, Finger.LEFT_MIDDLE, 1.1)
        self.set_key_info("4", 4, 0, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("5", 5, 0, 0, Finger.LEFT_INDEX, 1.2)
        self.set_key_info("6", 6, 0, 0, Finger.RIGHT_INDEX, 1.2)
        self.set_key_info("7", 7, 0, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("8", 8, 0, 0, Finger.RIGHT_MIDDLE, 1.1)
        self.set_key_info("9", 9, 0, 0, Finger.RIGHT_RING, 1.2)
        self.set_key_info("0", 10, 0, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("-", 11, 0, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("=", 12, 0, 0, Finger.RIGHT_PINKY, 1.5)

        # Middle row (y=1) - home row has lowest effort
        self.set_key_info("q", 0, 1, 0, Finger.LEFT_PINKY, 1.1)
        self.set_key_info("w", 1, 1, 0, Finger.LEFT_RING, 1.0)
        self.set_key_info("e", 2, 1, 0, Finger.LEFT_MIDDLE, 0.9)
        self.set_key_info("r", 3, 1, 0, Finger.LEFT_INDEX, 0.9)
        self.set_key_info("t", 4, 1, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("y", 5, 1, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info("u", 6, 1, 0, Finger.RIGHT_INDEX, 0.9)
        self.set_key_info("i", 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9)
        self.set_key_info("o", 8, 1, 0, Finger.RIGHT_RING, 1.0)
        self.set_key_info("p", 9, 1, 0, Finger.RIGHT_PINKY, 1.1)
        self.set_key_info("[", 10, 1, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("]", 11, 1, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("\\", 12, 1, 0, Finger.RIGHT_PINKY, 1.5)

        # Home row (y=2) - most comfortable
        self.set_key_info("a", 0, 2, 0, Finger.LEFT_PINKY, 0.8)
        self.set_key_info("s", 1, 2, 0, Finger.LEFT_RING, 0.7)
        self.set_key_info("d", 2, 2, 0, Finger.LEFT_MIDDLE, 0.6)
        self.set_key_info("f", 3, 2, 0, Finger.LEFT_INDEX, 0.6)
        self.set_key_info("g", 4, 2, 0, Finger.LEFT_INDEX, 0.7)
        self.set_key_info("h", 5, 2, 0, Finger.RIGHT_INDEX, 0.7)
        self.set_key_info("j", 6, 2, 0, Finger.RIGHT_INDEX, 0.6)
        self.set_key_info("k", 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6)
        self.set_key_info("l", 8, 2, 0, Finger.RIGHT_RING, 0.7)
        self.set_key_info(";", 9, 2, 0, Finger.RIGHT_PINKY, 0.8)
        self.set_key_info("'", 10, 2, 0, Finger.RIGHT_PINKY, 1.0)

        # Bottom row (y=3)
        self.set_key_info("z", 0, 3, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("x", 1, 3, 0, Finger.LEFT_RING, 1.1)
        self.set_key_info("c", 2, 3, 0, Finger.LEFT_MIDDLE, 1.0)
        self.set_key_info("v", 3, 3, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("b", 4, 3, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("n", 5, 3, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("m", 6, 3, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info(",", 7, 3, 0, Finger.RIGHT_MIDDLE, 1.0)
        self.set_key_info(".", 8, 3, 0, Finger.RIGHT_RING, 1.1)
        self.set_key_info("/", 9, 3, 0, Finger.RIGHT_PINKY, 1.2)

        # Special characters (shifted layer, z=1)
        self.set_key_info("~", 0, 0, 1, Finger.LEFT_PINKY, 1.5)
        shift_chars = "!@#$%^&*()_+"
        shift_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE, 
                        Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                        Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                        Finger.RIGHT_PINKY, Finger.RIGHT_PINKY, Finger.RIGHT_PINKY]

        for i, char in enumerate(shift_chars):
            self.set_key_info(char, i + 1, 0, 1, shift_fingers[i], 1.6)

        # Space bar
        self.set_key_info(" ", 3, 4, 0, Finger.RIGHT_THUMB, 0.5)

class DvorakLayout(KeyboardLayout):
    """Enhanced Dvorak keyboard layout with finger assignments."""

    def __init__(self):
        super().__init__("Dvorak")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0)
        self.set_key_info("`", 0, 0, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("1", 1, 0, 0, Finger.LEFT_PINKY, 1.3)
        self.set_key_info("2", 2, 0, 0, Finger.LEFT_RING, 1.2)
        self.set_key_info("3", 3, 0, 0, Finger.LEFT_MIDDLE, 1.1)
        self.set_key_info("4", 4, 0, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("5", 5, 0, 0, Finger.LEFT_INDEX, 1.2)
        self.set_key_info("6", 6, 0, 0, Finger.RIGHT_INDEX, 1.2)
        self.set_key_info("7", 7, 0, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("8", 8, 0, 0, Finger.RIGHT_MIDDLE, 1.1)
        self.set_key_info("9", 9, 0, 0, Finger.RIGHT_RING, 1.2)
        self.set_key_info("0", 10, 0, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("[", 11, 0, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("]", 12, 0, 0, Finger.RIGHT_PINKY, 1.5)

        # Middle row (y=1) - Dvorak vowels and consonants
        self.set_key_info("'", 0, 1, 0, Finger.LEFT_PINKY, 1.1)
        self.set_key_info(",", 1, 1, 0, Finger.LEFT_RING, 1.0)
        self.set_key_info(".", 2, 1, 0, Finger.LEFT_MIDDLE, 0.9)
        self.set_key_info("p", 3, 1, 0, Finger.LEFT_INDEX, 0.9)
        self.set_key_info("y", 4, 1, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("f", 5, 1, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info("g", 6, 1, 0, Finger.RIGHT_INDEX, 0.9)
        self.set_key_info("c", 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9)
        self.set_key_info("r", 8, 1, 0, Finger.RIGHT_RING, 1.0)
        self.set_key_info("l", 9, 1, 0, Finger.RIGHT_PINKY, 1.1)
        self.set_key_info("/", 10, 1, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("=", 11, 1, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("\\", 12, 1, 0, Finger.RIGHT_PINKY, 1.5)

        # Home row (y=2) - Dvorak home row optimization
        self.set_key_info("a", 0, 2, 0, Finger.LEFT_PINKY, 0.8)
        self.set_key_info("o", 1, 2, 0, Finger.LEFT_RING, 0.7)
        self.set_key_info("e", 2, 2, 0, Finger.LEFT_MIDDLE, 0.6)
        self.set_key_info("u", 3, 2, 0, Finger.LEFT_INDEX, 0.6)
        self.set_key_info("i", 4, 2, 0, Finger.LEFT_INDEX, 0.7)
        self.set_key_info("d", 5, 2, 0, Finger.RIGHT_INDEX, 0.7)
        self.set_key_info("h", 6, 2, 0, Finger.RIGHT_INDEX, 0.6)
        self.set_key_info("t", 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6)
        self.set_key_info("n", 8, 2, 0, Finger.RIGHT_RING, 0.7)
        self.set_key_info("s", 9, 2, 0, Finger.RIGHT_PINKY, 0.8)
        self.set_key_info("-", 10, 2, 0, Finger.RIGHT_PINKY, 1.0)

        # Bottom row (y=3)
        self.set_key_info(";", 0, 3, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("q", 1, 3, 0, Finger.LEFT_RING, 1.1)
        self.set_key_info("j", 2, 3, 0, Finger.LEFT_MIDDLE, 1.0)
        self.set_key_info("k", 3, 3, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("x", 4, 3, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("b", 5, 3, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("m", 6, 3, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info("w", 7, 3, 0, Finger.RIGHT_MIDDLE, 1.0)
        self.set_key_info("v", 8, 3, 0, Finger.RIGHT_RING, 1.1)
        self.set_key_info("z", 9, 3, 0, Finger.RIGHT_PINKY, 1.2)

        # Space bar
        self.set_key_info(" ", 3, 4, 0, Finger.RIGHT_THUMB, 0.5)

class ColemakLayout(KeyboardLayout):
    """Colemak keyboard layout with finger assignments."""

    def __init__(self):
        super().__init__("Colemak")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0) - same as QWERTY
        self.set_key_info("`", 0, 0, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("1", 1, 0, 0, Finger.LEFT_PINKY, 1.3)
        self.set_key_info("2", 2, 0, 0, Finger.LEFT_RING, 1.2)
        self.set_key_info("3", 3, 0, 0, Finger.LEFT_MIDDLE, 1.1)
        self.set_key_info("4", 4, 0, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("5", 5, 0, 0, Finger.LEFT_INDEX, 1.2)
        self.set_key_info("6", 6, 0, 0, Finger.RIGHT_INDEX, 1.2)
        self.set_key_info("7", 7, 0, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("8", 8, 0, 0, Finger.RIGHT_MIDDLE, 1.1)
        self.set_key_info("9", 9, 0, 0, Finger.RIGHT_RING, 1.2)
        self.set_key_info("0", 10, 0, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("-", 11, 0, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("=", 12, 0, 0, Finger.RIGHT_PINKY, 1.5)

        # Middle row (y=1) - Colemak arrangement
        self.set_key_info("q", 0, 1, 0, Finger.LEFT_PINKY, 1.1)
        self.set_key_info("w", 1, 1, 0, Finger.LEFT_RING, 1.0)
        self.set_key_info("f", 2, 1, 0, Finger.LEFT_MIDDLE, 0.9)
        self.set_key_info("p", 3, 1, 0, Finger.LEFT_INDEX, 0.9)
        self.set_key_info("g", 4, 1, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("j", 5, 1, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info("l", 6, 1, 0, Finger.RIGHT_INDEX, 0.9)
        self.set_key_info("u", 7, 1, 0, Finger.RIGHT_MIDDLE, 0.9)
        self.set_key_info("y", 8, 1, 0, Finger.RIGHT_RING, 1.0)
        self.set_key_info(";", 9, 1, 0, Finger.RIGHT_PINKY, 1.1)
        self.set_key_info("[", 10, 1, 0, Finger.RIGHT_PINKY, 1.3)
        self.set_key_info("]", 11, 1, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("\\", 12, 1, 0, Finger.RIGHT_PINKY, 1.5)

        # Home row (y=2) - Colemak optimization
        self.set_key_info("a", 0, 2, 0, Finger.LEFT_PINKY, 0.8)
        self.set_key_info("r", 1, 2, 0, Finger.LEFT_RING, 0.7)
        self.set_key_info("s", 2, 2, 0, Finger.LEFT_MIDDLE, 0.6)
        self.set_key_info("t", 3, 2, 0, Finger.LEFT_INDEX, 0.6)
        self.set_key_info("d", 4, 2, 0, Finger.LEFT_INDEX, 0.7)
        self.set_key_info("h", 5, 2, 0, Finger.RIGHT_INDEX, 0.7)
        self.set_key_info("n", 6, 2, 0, Finger.RIGHT_INDEX, 0.6)
        self.set_key_info("e", 7, 2, 0, Finger.RIGHT_MIDDLE, 0.6)
        self.set_key_info("i", 8, 2, 0, Finger.RIGHT_RING, 0.7)
        self.set_key_info("o", 9, 2, 0, Finger.RIGHT_PINKY, 0.8)
        self.set_key_info("'", 10, 2, 0, Finger.RIGHT_PINKY, 1.0)

        # Bottom row (y=3) - Colemak arrangement
        self.set_key_info("z", 0, 3, 0, Finger.LEFT_PINKY, 1.2)
        self.set_key_info("x", 1, 3, 0, Finger.LEFT_RING, 1.1)
        self.set_key_info("c", 2, 3, 0, Finger.LEFT_MIDDLE, 1.0)
        self.set_key_info("v", 3, 3, 0, Finger.LEFT_INDEX, 1.0)
        self.set_key_info("b", 4, 3, 0, Finger.LEFT_INDEX, 1.1)
        self.set_key_info("k", 5, 3, 0, Finger.RIGHT_INDEX, 1.1)
        self.set_key_info("m", 6, 3, 0, Finger.RIGHT_INDEX, 1.0)
        self.set_key_info(",", 7, 3, 0, Finger.RIGHT_MIDDLE, 1.0)
        self.set_key_info(".", 8, 3, 0, Finger.RIGHT_RING, 1.1)
        self.set_key_info("/", 9, 3, 0, Finger.RIGHT_PINKY, 1.2)

        # Space bar
        self.set_key_info(" ", 3, 4, 0, Finger.RIGHT_THUMB, 0.5)

class WorkmanLayout(KeyboardLayout):
    """Workman keyboard layout with finger assignments."""

    def __init__(self):
        super().__init__("Workman")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0) - same as QWERTY
        self.set_key_info("`", 0, 0, 0, Finger.LEFT_PINKY, 1.2)
        for i in range(10):
            self.set_key_info(str(i+1)[-1], i + 1, 0, 0, 
                            [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE, 
                             Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                             Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                             Finger.RIGHT_PINKY][i], 1.1 + i * 0.05)
        self.set_key_info("-", 11, 0, 0, Finger.RIGHT_PINKY, 1.4)
        self.set_key_info("=", 12, 0, 0, Finger.RIGHT_PINKY, 1.5)

        # Middle row (y=1) - Workman arrangement
        workman_top = "qdrwbjfup;"
        for i, char in enumerate(workman_top):
            finger = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE, 
                     Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                     Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                     Finger.RIGHT_PINKY][i]
            self.set_key_info(char, i, 1, 0, finger, 0.9 + i * 0.02)

        # Home row (y=2) - Workman optimization  
        workman_home = "ashtgyneoi"
        for i, char in enumerate(workman_home):
            finger = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE, 
                     Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                     Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                     Finger.RIGHT_PINKY][i]
            self.set_key_info(char, i, 2, 0, finger, 0.6 + i * 0.02)

        # Bottom row (y=3)
        workman_bottom = "zxmcvkl,./"
        for i, char in enumerate(workman_bottom):
            finger = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE, 
                     Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                     Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                     Finger.RIGHT_PINKY][i]
            self.set_key_info(char, i, 3, 0, finger, 1.0 + i * 0.02)

        # Space bar
        self.set_key_info(" ", 3, 4, 0, Finger.RIGHT_THUMB, 0.5)

def calculate_distance(point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
    """Calculate the Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def calculate_hand_alternation_rate(text: str, layout: KeyboardLayout) -> float:
    """Calculate the percentage of hand alternations in typing."""
    hand_sequence = layout.get_hand_sequence(text)
    if len(hand_sequence) <= 1:
        return 0.0

    alternations = sum(1 for i in range(1, len(hand_sequence)) 
                      if hand_sequence[i] != hand_sequence[i-1])
    return (alternations / (len(hand_sequence) - 1)) * 100

def calculate_finger_utilization(text: str, layout: KeyboardLayout) -> Dict[Finger, int]:
    """Calculate how many times each finger is used."""
    finger_sequence = layout.get_finger_sequence(text)
    return dict(Counter(finger_sequence))

def calculate_same_finger_transitions(text: str, layout: KeyboardLayout) -> Tuple[int, float]:
    """Calculate same-finger transitions (awkward typing)."""
    finger_sequence = layout.get_finger_sequence(text)
    if len(finger_sequence) <= 1:
        return 0, 0.0

    same_finger = sum(1 for i in range(1, len(finger_sequence)) 
                     if finger_sequence[i] == finger_sequence[i-1])
    percentage = (same_finger / (len(finger_sequence) - 1)) * 100
    return same_finger, percentage

def calculate_bigram_efficiency(text: str, layout: KeyboardLayout) -> float:
    """Calculate efficiency of bigram (two-character) combinations."""
    if len(text) < 2:
        return 0.0

    bigrams = [text[i:i+2].lower() for i in range(len(text) - 1)]
    total_efficiency = 0.0
    valid_bigrams = 0

    for bigram in bigrams:
        if len(bigram) == 2:
            char1, char2 = bigram
            info1 = layout.get_key_info(char1)
            info2 = layout.get_key_info(char2)

            if info1 and info2:
                # Calculate efficiency based on hand alternation and finger usage
                efficiency = 1.0

                # Bonus for hand alternation
                if info1.hand != info2.hand:
                    efficiency *= 1.2

                # Penalty for same finger usage
                if info1.finger == info2.finger:
                    efficiency *= 0.3

                # Penalty based on distance
                distance = calculate_distance(info1.position, info2.position)
                efficiency *= max(0.1, 1.0 - (distance / 10.0))

                # Factor in key effort
                efficiency *= (2.0 - info1.effort) * (2.0 - info2.effort) / 4.0

                total_efficiency += efficiency
                valid_bigrams += 1

    return total_efficiency / max(1, valid_bigrams)

def calculate_trigram_efficiency(text: str, layout: KeyboardLayout) -> float:
    """Calculate efficiency of trigram (three-character) combinations."""
    if len(text) < 3:
        return 0.0

    trigrams = [text[i:i+3].lower() for i in range(len(text) - 2)]
    total_efficiency = 0.0
    valid_trigrams = 0

    for trigram in trigrams:
        if len(trigram) == 3:
            char1, char2, char3 = trigram
            info1 = layout.get_key_info(char1)
            info2 = layout.get_key_info(char2)
            info3 = layout.get_key_info(char3)

            if info1 and info2 and info3:
                efficiency = 1.0

                # Check for alternating hands (ideal pattern)
                hands = [info1.hand, info2.hand, info3.hand]
                if hands[0] != hands[1] and hands[1] != hands[2]:
                    efficiency *= 1.3
                elif hands[0] == hands[1] == hands[2]:
                    efficiency *= 0.7

                # Check for same finger usage (very bad)
                fingers = [info1.finger, info2.finger, info3.finger]
                same_finger_count = len(set(fingers))
                if same_finger_count == 1:
                    efficiency *= 0.2
                elif same_finger_count == 2:
                    efficiency *= 0.6

                # Factor in total path distance
                dist1 = calculate_distance(info1.position, info2.position)
                dist2 = calculate_distance(info2.position, info3.position)
                total_distance = dist1 + dist2
                efficiency *= max(0.1, 1.0 - (total_distance / 15.0))

                total_efficiency += efficiency
                valid_trigrams += 1

    return total_efficiency / max(1, valid_trigrams)

def calculate_comprehensive_stats(text: str, layout: KeyboardLayout) -> TypingStats:
    """Calculate comprehensive typing statistics for a text."""
    # Basic distance calculations
    path = layout.get_3d_path(text)
    total_distance = 0.0
    if len(path) > 1:
        for i in range(1, len(path)):
            total_distance += calculate_distance(path[i-1], path[i])

    avg_distance_per_char = total_distance / max(1, len(text))

    # Same row transitions
    same_row_transitions = 0
    for i in range(1, len(path)):
        if abs(path[i][1] - path[i-1][1]) < 0.1:
            same_row_transitions += 1

    same_row_percentage = (same_row_transitions / max(1, len(path) - 1)) * 100

    # Hand alternation
    hand_alternation_rate = calculate_hand_alternation_rate(text, layout)

    # Finger utilization
    finger_utilization = calculate_finger_utilization(text, layout)
    most_used_fingers = sorted(finger_utilization.items(), key=lambda x: x[1], reverse=True)[:3]

    # Effort calculation
    total_effort = 0.0
    char_count = 0
    for char in text.lower():
        info = layout.get_key_info(char)
        if info:
            total_effort += info.effort
            char_count += 1

    avg_effort_per_char = total_effort / max(1, char_count)

    # Same finger transitions
    same_finger_transitions, same_finger_percentage = calculate_same_finger_transitions(text, layout)

    # Bigram and trigram efficiency
    bigram_efficiency = calculate_bigram_efficiency(text, layout)
    trigram_efficiency = calculate_trigram_efficiency(text, layout)

    return TypingStats(
        total_distance=total_distance,
        avg_distance_per_char=avg_distance_per_char,
        same_row_transitions=same_row_transitions,
        same_row_percentage=same_row_percentage,
        hand_alternation_rate=hand_alternation_rate,
        finger_utilization=finger_utilization,
        most_used_fingers=most_used_fingers,
        total_effort=total_effort,
        avg_effort_per_char=avg_effort_per_char,
        same_finger_transitions=same_finger_transitions,
        same_finger_percentage=same_finger_percentage,
        bigram_efficiency=bigram_efficiency,
        trigram_efficiency=trigram_efficiency
    )

def calculate_statistical_analysis(values: List[float]) -> StatisticalAnalysis:
    """Calculate comprehensive statistical analysis of a dataset."""
    if not values:
        return StatisticalAnalysis(0, 0, 0, 0, 0, 0, (0, 0))

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

    # 95% confidence interval (approximate)
    if len(values) > 1:
        margin_error = 1.96 * (std_dev / math.sqrt(len(values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
    else:
        ci_lower = ci_upper = mean_val

    return StatisticalAnalysis(
        mean=mean_val,
        median=median_val,
        std_dev=std_dev,
        variance=variance,
        min_value=min_val,
        max_value=max_val,
        confidence_interval_95=(ci_lower, ci_upper)
    )

def generate_word_path(word: str, layout: KeyboardLayout) -> List[Tuple[float, float, float]]:
    """Return the 3D path for a word on a specific keyboard layout."""
    return layout.get_3d_path(word)

def project_orthogonal_3d_path(path: List[Tuple[float, float, float]]) -> Dict[str, List[Tuple[float, float]]]:
    """Project a 3D path onto 6 orthogonal directions (2D/3D projections)."""
    front = [(x, y) for x, y, z in path]  # x-y plane
    back = [(x, y) for x, y, z in path if z > 0]  # x-y with z>0
    left = [(y, z) for x, y, z in path]  # y-z plane
    right = [(y, z) for x, y, z in path if x > 5]  # y-z with x>5
    top = [(x, z) for x, y, z in path]  # x-z plane
    bottom = [(x, z) for x, y, z in path if y > 2]  # x-z with y>2
    return {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }

def to_hex(projection: List[Tuple[float, float]]) -> List[str]:
    """Convert 2D coordinates to hex code (4 hex digits per point)."""
    hex_code = []
    for point in projection:
        hex_digits = ""
        for axis in point:
            # Convert to 2 hex digits (00-FF)
            hex_axis = hex(int(abs(axis)) % 256)[2:].upper().zfill(2)
            hex_digits += hex_axis
        hex_code.append(hex_digits)
    return hex_code

def compress_path(path: List[Tuple[float, float, float]]) -> bytes:
    """Compress a 3D path using zlib."""
    return zlib.compress(json.dumps(path).encode('utf-8'))

def decompress_path(compressed_path: bytes) -> List[Tuple[float, float, float]]:
    """Decompress a 3D path using zlib."""
    json_data = zlib.decompress(compressed_path).decode('utf-8')
    # Convert the list of lists back to list of tuples
    data = json.loads(json_data)
    return [tuple(point) for point in data]

def gzip_compress_path(path: List[Tuple[float, float, float]]) -> bytes:
    """Compress a 3D path using gzip (corrected version)."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(json.dumps(path).encode('utf-8'))
    return buffer.getvalue()

def gzip_decompress_path(compressed_path: bytes) -> List[Tuple[float, float, float]]:
    """Decompress a 3D path using gzip (corrected version)."""
    buffer = io.BytesIO(compressed_path)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
        json_data = f.read().decode('utf-8')
        # Convert the list of lists back to list of tuples
        data = json.loads(json_data)
        return [tuple(point) for point in data]

def compare_layout_efficiency(text: str, layouts: List[KeyboardLayout]) -> Dict[str, TypingStats]:
    """Compare the comprehensive efficiency of multiple keyboard layouts."""
    results = {}
    for layout in layouts:
        results[layout.layout_name] = calculate_comprehensive_stats(text, layout)
    return results

def analyze_multiple_texts(texts: List[str], layouts: List[KeyboardLayout]) -> Dict[str, Dict[str, StatisticalAnalysis]]:
    """Analyze multiple texts across layouts and provide statistical summaries."""
    layout_metrics = {layout.layout_name: {
        'total_distance': [],
        'avg_distance_per_char': [],
        'hand_alternation_rate': [],
        'avg_effort_per_char': [],
        'same_finger_percentage': [],
        'bigram_efficiency': [],
        'trigram_efficiency': []
    } for layout in layouts}

    for text in texts:
        results = compare_layout_efficiency(text, layouts)
        for layout_name, stats in results.items():
            layout_metrics[layout_name]['total_distance'].append(stats.total_distance)
            layout_metrics[layout_name]['avg_distance_per_char'].append(stats.avg_distance_per_char)
            layout_metrics[layout_name]['hand_alternation_rate'].append(stats.hand_alternation_rate)
            layout_metrics[layout_name]['avg_effort_per_char'].append(stats.avg_effort_per_char)
            layout_metrics[layout_name]['same_finger_percentage'].append(stats.same_finger_percentage)
            layout_metrics[layout_name]['bigram_efficiency'].append(stats.bigram_efficiency)
            layout_metrics[layout_name]['trigram_efficiency'].append(stats.trigram_efficiency)

    # Calculate statistical analysis for each metric
    statistical_results = {}
    for layout_name, metrics in layout_metrics.items():
        statistical_results[layout_name] = {}
        for metric_name, values in metrics.items():
            statistical_results[layout_name][metric_name] = calculate_statistical_analysis(values)

    return statistical_results

def display_comprehensive_results(stats: TypingStats, layout_name: str) -> None:
    """Display comprehensive typing statistics."""
    print(f"\n=== {layout_name} Layout Analysis ===")
    print(f"Total Distance: {stats.total_distance:.2f}")
    print(f"Avg Distance per Character: {stats.avg_distance_per_char:.3f}")
    print(f"Hand Alternation Rate: {stats.hand_alternation_rate:.1f}%")
    print(f"Total Effort: {stats.total_effort:.2f}")
    print(f"Avg Effort per Character: {stats.avg_effort_per_char:.3f}")
    print(f"Same Finger Transitions: {stats.same_finger_transitions} ({stats.same_finger_percentage:.1f}%)")
    print(f"Same Row Transitions: {stats.same_row_transitions} ({stats.same_row_percentage:.1f}%)")
    print(f"Bigram Efficiency: {stats.bigram_efficiency:.3f}")
    print(f"Trigram Efficiency: {stats.trigram_efficiency:.3f}")

    print(f"\nMost Used Fingers:")
    for finger, count in stats.most_used_fingers:
        print(f"  {finger.value}: {count} times")

def display_statistical_summary(results: Dict[str, Dict[str, StatisticalAnalysis]]) -> None:
    """Display statistical analysis across multiple texts."""
    print("\n=== Statistical Analysis Summary ===")

    metrics = ['total_distance', 'avg_distance_per_char', 'hand_alternation_rate', 
              'avg_effort_per_char', 'same_finger_percentage', 'bigram_efficiency', 'trigram_efficiency']

    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"{'Layout':<12} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)

        for layout_name, layout_stats in results.items():
            stats = layout_stats[metric]
            print(f"{layout_name:<12} {stats.mean:<8.3f} {stats.median:<8.3f} "
                  f"{stats.std_dev:<8.3f} {stats.min_value:<8.3f} {stats.max_value:<8.3f}")

def find_optimal_layout(results: Dict[str, Dict[str, StatisticalAnalysis]]) -> str:
    """Determine the optimal layout based on multiple weighted criteria."""
    layout_scores = {}

    # Weight factors for different metrics (higher is better)
    weights = {
        'total_distance': -1.0,  # Lower is better
        'avg_distance_per_char': -1.0,  # Lower is better
        'hand_alternation_rate': 1.0,  # Higher is better
        'avg_effort_per_char': -1.0,  # Lower is better
        'same_finger_percentage': -2.0,  # Lower is much better
        'bigram_efficiency': 1.5,  # Higher is better
        'trigram_efficiency': 1.5  # Higher is better
    }

    for layout_name, layout_stats in results.items():
        score = 0.0
        for metric, weight in weights.items():
            # Use mean value for scoring
            metric_value = layout_stats[metric].mean
            score += weight * metric_value
        layout_scores[layout_name] = score

    return max(layout_scores.items(), key=lambda x: x[1])

def analyze_finger_workload_distribution(texts: List[str], layout: KeyboardLayout) -> Dict[Finger, StatisticalAnalysis]:
    """Analyze workload distribution across fingers."""
    finger_usage_per_text = defaultdict(list)

    for text in texts:
        finger_utilization = calculate_finger_utilization(text, layout)
        total_chars = sum(finger_utilization.values())

        # Calculate percentage usage for each finger
        for finger in Finger:
            usage_count = finger_utilization.get(finger, 0)
            usage_percentage = (usage_count / max(1, total_chars)) * 100
            finger_usage_per_text[finger].append(usage_percentage)

    # Calculate statistical analysis for each finger
    finger_stats = {}
    for finger, usage_list in finger_usage_per_text.items():
        finger_stats[finger] = calculate_statistical_analysis(usage_list)

    return finger_stats

def example_comprehensive_analysis() -> None:
    """Run a comprehensive analysis example with multiple layouts and texts."""
    # Create all layouts
    layouts = [
        QWERTYLayout(),
        DvorakLayout(),
        ColemakLayout(),
        WorkmanLayout()
    ]

    # Extended sample texts for better statistical analysis
    sample_texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "programming is fun and challenging",
        "efficiency analysis of keyboard layouts",
        "python programming language features",
        "comprehensive keyboard layout comparison study",
        "statistical analysis and data visualization",
        "machine learning algorithms and applications",
        "software engineering best practices guide",
        "user interface design principles methodology",
        "database management systems optimization",
        "web development frameworks comparison analysis"
    ]

    print("=== COMPREHENSIVE KEYBOARD LAYOUT ANALYSIS ===\n")

    # Analyze individual text samples
    print("Individual Text Analysis:")
    for i, text in enumerate(sample_texts[:3]):  # Show first 3 for brevity
        print(f"\nText {i+1}: '{text}'")
        results = compare_layout_efficiency(text, layouts)

        best_layout = min(results.items(), key=lambda x: x[1].total_distance)
        worst_layout = max(results.items(), key=lambda x: x[1].total_distance)

        print(f"Best layout: {best_layout[0]} (distance: {best_layout[1].total_distance:.2f})")
        print(f"Worst layout: {worst_layout[0]} (distance: {worst_layout[1].total_distance:.2f})")

        improvement = ((worst_layout[1].total_distance - best_layout[1].total_distance) / 
                      worst_layout[1].total_distance) * 100
        print(f"Improvement: {improvement:.1f}%")

    # Statistical analysis across all texts
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS ACROSS ALL TEXTS")
    print(f"{'='*60}")

    statistical_results = analyze_multiple_texts(sample_texts, layouts)
    display_statistical_summary(statistical_results)

    # Find optimal layout
    optimal_layout, optimal_score = find_optimal_layout(statistical_results)
    print(f"\n=== OPTIMAL LAYOUT RECOMMENDATION ===")
    print(f"Recommended Layout: {optimal_layout}")
    print(f"Composite Score: {optimal_score:.3f}")

    # Finger workload analysis
    print(f"\n=== FINGER WORKLOAD ANALYSIS ===")
    for layout in layouts:
        print(f"\n{layout.layout_name} Finger Usage Distribution:")
        finger_stats = analyze_finger_workload_distribution(sample_texts, layout)

        # Sort by mean usage
        sorted_fingers = sorted(finger_stats.items(), key=lambda x: x[1].mean, reverse=True)

        print(f"{'Finger':<15} {'Mean %':<8} {'Std Dev':<8} {'Min %':<8} {'Max %':<8}")
        print("-" * 55)
        for finger, stats in sorted_fingers[:5]:  # Top 5 most used
            print(f"{finger.value:<15} {stats.mean:<8.1f} {stats.std_dev:<8.1f} "
                  f"{stats.min_value:<8.1f} {stats.max_value:<8.1f}")

    # Compression demonstration
    print(f"\n=== COMPRESSION ANALYSIS ===")
    test_word = "comprehensive"
    test_layout = layouts[0]  # QWERTY
    path = test_layout.get_3d_path(test_word)

    original_size = len(json.dumps(path).encode('utf-8'))
    zlib_compressed = compress_path(path)
    gzip_compressed = gzip_compress_path(path)

    print(f"Original path size: {original_size} bytes")
    print(f"Zlib compressed: {len(zlib_compressed)} bytes ({(len(zlib_compressed)/original_size)*100:.1f}%)")
    print(f"Gzip compressed: {len(gzip_compressed)} bytes ({(len(gzip_compressed)/original_size)*100:.1f}%)")

    # Verify decompression works
    decompressed_zlib = decompress_path(zlib_compressed)
    decompressed_gzip = gzip_decompress_path(gzip_compressed)

    print(f"Zlib decompression successful: {decompressed_zlib == path}")
    print(f"Gzip decompression successful: {decompressed_gzip == path}")

def main() -> None:
    """Run the enhanced keyboard layout analysis."""
    example_comprehensive_analysis()

    # Example of single word analysis with hex codes
    print(f"\n{'='*60}")
    print("HEX CODE GENERATION EXAMPLE")
    print(f"{'='*60}")

    qwerty = QWERTYLayout()
    test_word = "hello"

    print(f"Analyzing word '{test_word}' on QWERTY layout:")
    path = generate_word_path(test_word, qwerty)
    print(f"3D Path: {path}")

    projections = project_orthogonal_3d_path(path)
    for view, projection in projections.items():
        if projection:  # Only show non-empty projections
            hex_code = to_hex(projection)
            print(f"{view.capitalize()} View: {projection}")
            print(f"{view.capitalize()} Hex: {' '.join(hex_code)}")

if __name__ == "__main__":
    main()

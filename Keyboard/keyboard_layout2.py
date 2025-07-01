
# keyboard_layout.py

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Tuple, Optional

class KeyboardLayout(ABC):
    """Base class for keyboard layouts with 3D coordinate mapping."""

    def __init__(self, layout_name: str):
        """Initialize a keyboard layout with a name."""
        self.layout_name = layout_name
        self.key_positions: Dict[str, Tuple[float, float, float]] = {}

    @abstractmethod
    def set_key_positions(self) -> None:
        """Set key positions for the keyboard layout."""
        pass

    def set_key_position(self, char: str, x: float, y: float, z: float = 0) -> None:
        """Assign 3D coordinates to a key on the keyboard layout."""
        self.key_positions[char.lower()] = (x, y, z)

    def get_key_position(self, char: str) -> Tuple[float, float, float]:
        """Get the 3D coordinates for a key."""
        return self.key_positions.get(char.lower())

    def get_3d_path(self, word: str) -> List[Tuple[float, float, float]]:
        """Return a list of 3D coordinates for each character in the word."""
        path = []
        for char in word.lower():
            if char in self.key_positions:
                path.append(self.key_positions[char])
        return path


class QWERTYLayout(KeyboardLayout):
    """QWERTY keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("QWERTY")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0)
        self.set_key_position("`", 0, 0, 0)
        for i, char in enumerate("1234567890-="):
            self.set_key_position(char, i + 1, 0, 0)

        # Middle row (y=1)
        for i, char in enumerate("qwertyuiop[]\\"):
            self.set_key_position(char, i, 1, 0)

        # Lower row (y=2)
        for i, char in enumerate("asdfghjkl;'"):
            self.set_key_position(char, i, 2, 0)

        # Bottom row (y=3)
        for i, char in enumerate("zxcvbnm,./"):
            self.set_key_position(char, i, 3, 0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("~", 0, 0, 1)
        for i, char in enumerate("!@#$%^&*()_+"):
            self.set_key_position(char, i + 1, 0, 1)

        # Space bar
        self.set_key_position(" ", 3, 4, 0)


class DvorakLayout(KeyboardLayout):
    """Dvorak keyboard layout with 3D coordinates."""

    def __init__(self):
        super().__init__("Dvorak")
        self.set_key_positions()

    def set_key_positions(self) -> None:
        # Top row (y=0)
        self.set_key_position("`", 0, 0, 0)
        for i, char in enumerate("1234567890[]"):
            self.set_key_position(char, i + 1, 0, 0)

        # Middle row (y=1)
        for i, char in enumerate("',.pyfgcrl/=\\"):
            self.set_key_position(char, i, 1, 0)

        # Lower row (y=2)
        for i, char in enumerate("aoeuidhtns-"):
            self.set_key_position(char, i, 2, 0)

        # Bottom row (y=3)
        for i, char in enumerate(";qjkxbmwvz"):
            self.set_key_position(char, i, 3, 0)

        # Special characters (shifted layer, z=1)
        self.set_key_position("~", 0, 0, 1)
        for i, char in enumerate("!@#$%^&*(){}"):
            self.set_key_position(char, i + 1, 0, 1)

        # Space bar
        self.set_key_position(" ", 3, 4, 0)


def generate_word_path(word: str, layout: KeyboardLayout) -> List[Tuple[float, float, float]]:
    """Return the 3D path for a word on a specific keyboard layout."""
    return layout.get_3d_path(word)


def project_orthogonal_3d_path(path: List[Tuple[float, float, float]]) -> Dict[str, List[Tuple[float, float]]]:
    """Project a 3D path onto 6 orthogonal directions (2D/3D projections)."""
    front = [(x, y) for x, y, z in path]  # x-y plane
    back = [(x, y, 1) for x, y, z in path]  # x-y with z=1
    left = [(y, z) for x, y, z in path]  # y-z plane
    right = [(y, z, 10) for x, y, z in path]  # y-z with x=10
    top = [(x, z) for x, y, z in path]  # x-z plane
    bottom = [(x, z, 10) for x, y, z in path]  # x-z with y=10
    return {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def to_hex(projection: List[Tuple[float, float]]) -> List[str]:
    """Convert 2D/3D coordinates to hex code (4 or 6 hex digits per point)."""
    hex_code = []
    for point in projection:
        hex_digits = ""
        for axis in point:
            # Convert integer to 2 hex digits (00-FF)
            hex_axis = hex(int(axis))[2:].upper().zfill(2)
            hex_digits += hex_axis
        hex_code.append(hex_digits)
    return hex_code


def calculate_distance(point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
    """Calculate the Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def calculate_word_distance(word: str, layout: KeyboardLayout) -> float:
    """Calculate the total distance traveled when typing a word on a specific layout."""
    path = generate_word_path(word, layout)
    if len(path) <= 1:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += calculate_distance(path[i-1], path[i])

    return total_distance


def analyze_layout_efficiency(text: str, layout: KeyboardLayout) -> Dict[str, float]:
    """Analyze the efficiency of a keyboard layout for typing a specific text."""
    # Calculate metrics
    total_distance = calculate_word_distance(text, layout)
    avg_distance_per_char = total_distance / max(1, len(text))

    # Count same-finger transitions (simplified model)
    path = generate_word_path(text, layout)
    same_row_transitions = 0
    for i in range(1, len(path)):
        # Check if keys are in the same row (y-coordinate)
        if abs(path[i][1] - path[i-1][1]) < 0.1:
            same_row_transitions += 1

    same_row_percentage = (same_row_transitions / max(1, len(path) - 1)) * 100

    return {
        "total_distance": total_distance,
        "avg_distance_per_char": avg_distance_per_char,
        "same_row_transitions": same_row_transitions,
        "same_row_percentage": same_row_percentage
    }


def compare_layout_efficiency(text: str, layouts: List[KeyboardLayout]) -> Dict[str, Dict[str, float]]:
    """Compare the efficiency of multiple keyboard layouts for typing a specific text."""
    results = {}

    for layout in layouts:
        results[layout.layout_name] = analyze_layout_efficiency(text, layout)

    return results


def example_word_analysis(word: str, layout: KeyboardLayout) -> None:
    """Analyze a word's path on a keyboard layout and generate hex codes."""
    print(f"Analyzing word '{word}' on {layout.layout_name} layout:")

    # Generate 3D path
    path = generate_word_path(word, layout)
    print(f"3D Path: {path}")

    # Project to 6 orthogonal directions
    projections = project_orthogonal_3d_path(path)

    # Generate hex codes for all 6 views
    for view, projection in projections.items():
        hex_code = to_hex(projection)
        print(f"{view.capitalize()} View Hex Code: {hex_code}")


def display_efficiency_results(results: Dict[str, Dict[str, float]]) -> None:
    """Display efficiency comparison results in a readable format."""
    print("\n=== Keyboard Layout Efficiency Comparison ===")

    # Get all metric names from the first layout's results
    if not results:
        print("No results to display.")
        return

    first_layout = next(iter(results.values()))
    metrics = first_layout.keys()

    # Print header
    print(f"{'Metric':<25} | " + " | ".join(f"{layout:<15}" for layout in results.keys()))
    print("-" * 25 + "+" + "+".join(["-" * 17 for _ in results.keys()]))

    # Print each metric
    for metric in metrics:
        metric_display = metric.replace("_", " ").title()
        print(f"{metric_display:<25} | " + " | ".join(f"{results[layout][metric]:<15.2f}" for layout in results.keys()))


def summarize_layout_comparison(layout_stats: Dict[str, Dict[str, float]]) -> None:
    """Provide a summary of layout comparison with overall recommendations."""
    print("\n=== Overall Layout Efficiency Summary ===")

    # Calculate average metrics across all texts
    avg_metrics = {}
    for layout_name, stats in layout_stats.items():
        avg_metrics[layout_name] = {
            "avg_total_distance": sum(text_stats["total_distance"] for text_stats in stats.values()) / len(stats),
            "avg_distance_per_char": sum(text_stats["avg_distance_per_char"] for text_stats in stats.values()) / len(stats),
            "win_count": sum(1 for text_stats in stats.values() if text_stats["is_winner"]),
            "win_percentage": (sum(1 for text_stats in stats.values() if text_stats["is_winner"]) / len(stats)) * 100
        }

    # Display summary
    print(f"{'Metric':<25} | " + " | ".join(f"{layout:<15}" for layout in avg_metrics.keys()))
    print("-" * 25 + "+" + "+".join(["-" * 17 for _ in avg_metrics.keys()]))

    metrics_to_display = ["avg_total_distance", "avg_distance_per_char", "win_count", "win_percentage"]
    for metric in metrics_to_display:
        metric_display = metric.replace("_", " ").title()
        print(f"{metric_display:<25} | " + " | ".join(f"{avg_metrics[layout][metric]:<15.2f}" for layout in avg_metrics.keys()))

    # Determine overall recommendation
    best_layout = max(avg_metrics.items(), key=lambda x: x[1]["win_percentage"])[0]
    win_percentage = avg_metrics[best_layout]["win_percentage"]

    print(f"\nOverall Recommendation: {best_layout} layout")
    print(f"This layout was more efficient in {win_percentage:.1f}% of the tested texts.")

    # Additional insights
    if win_percentage < 70:
        print("\nNote: The efficiency difference between layouts is not overwhelming.")
        print("Consider your specific typing needs and the types of text you work with most frequently.")


def main() -> None:
    # Create layouts
    qwerty = QWERTYLayout()
    dvorak = DvorakLayout()

    # Example words and phrases for analysis
    sample_texts = [
        "hello",
        "the quick brown fox",
        "programming is fun",
        "efficiency analysis",
        "python programming",
        "keyboard layout comparison"
    ]

    # Analyze paths and hex codes for a simple word
    example_word_analysis("hello", qwerty)
    example_word_analysis("hello", dvorak)

    print("\n" + "="*50 + "\n")

    # Store results for summary
    layout_stats = {
        "QWERTY": {},
        "Dvorak": {}
    }

    # Compare layout efficiency for different texts
    for text in sample_texts:
        print(f"\nEfficiency analysis for: '{text}'")
        results = compare_layout_efficiency(text, [qwerty, dvorak])
        display_efficiency_results(results)

        # Calculate which layout is more efficient for this text
        qwerty_distance = results["QWERTY"]["total_distance"]
        dvorak_distance = results["Dvorak"]["total_distance"]

        # Store results for summary
        layout_stats["QWERTY"][text] = results["QWERTY"].copy()
        layout_stats["Dvorak"][text] = results["Dvorak"].copy()

        if qwerty_distance < dvorak_distance:
            efficiency_diff = ((dvorak_distance - qwerty_distance) / dvorak_distance) * 100
            print(f"\nQWERTY is {efficiency_diff:.2f}% more efficient for typing '{text}'")
            layout_stats["QWERTY"][text]["is_winner"] = True
            layout_stats["Dvorak"][text]["is_winner"] = False
        elif dvorak_distance < qwerty_distance:
            efficiency_diff = ((qwerty_distance - dvorak_distance) / qwerty_distance) * 100
            print(f"\nDvorak is {efficiency_diff:.2f}% more efficient for typing '{text}'")
            layout_stats["QWERTY"][text]["is_winner"] = False
            layout_stats["Dvorak"][text]["is_winner"] = True
        else:
            print("\nBoth layouts have equal efficiency for this text.")
            layout_stats["QWERTY"][text]["is_winner"] = False
            layout_stats["Dvorak"][text]["is_winner"] = False

    # Provide overall summary and recommendations
    summarize_layout_comparison(layout_stats)


if __name__ == "__main__":
    main()

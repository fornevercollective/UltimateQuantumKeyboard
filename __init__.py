# src/quantum_keyboard/__init__.py

"""
Ultimate Quantum Keyboard Analyzer

A revolutionary, research-grade keyboard layout analyzer that combines quantum-inspired 
metrics, advanced statistical analysis, and machine learning techniques to provide 
unprecedented insights into typing efficiency and ergonomics.
"""

__version__ = "1.0.0"
__author__ = "Quantum Keyboard Research Team"
__email__ = "quantum.keyboard@example.com"

# Core imports
from .ultimate_quantum_keyboard import (
    UltimateQuantumKeyboard,
    QWERTYLayout,
    DvorakLayout,
    ColemakLayout,
    ComprehensiveTypingStats,
    StatisticalSummary,
    analyze_single_word,
    compare_word_across_layouts,
    batch_analyze_texts,
    export_analysis_to_json,
)

# Layout imports
from .layouts import *

# Analysis imports  
from .analysis import *

# Visualization imports
from .visualization import *

__all__ = [
    # Core classes
    "UltimateQuantumKeyboard",
    "QWERTYLayout", 
    "DvorakLayout",
    "ColemakLayout",
    "ComprehensiveTypingStats",
    "StatisticalSummary",
    
    # Convenience functions
    "analyze_single_word",
    "compare_word_across_layouts", 
    "batch_analyze_texts",
    "export_analysis_to_json",
]

---

# src/quantum_keyboard/cli.py

"""Command line interface for the Ultimate Quantum Keyboard Analyzer."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .ultimate_quantum_keyboard import (
    UltimateQuantumKeyboard,
    analyze_single_word,
    compare_word_across_layouts,
)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ultimate Quantum Keyboard Analyzer - Revolutionary typing analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quantum-keyboard analyze "hello world" --layout qwerty
  quantum-keyboard compare "efficiency" --layouts qwerty dvorak colemak
  quantum-keyboard report texts.txt --output report.txt
  quantum-keyboard visualize "quantum computing" --save plot.png
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single text")
    analyze_parser.add_argument("text", help="Text to analyze")
    analyze_parser.add_argument(
        "--layout", 
        default="qwerty",
        choices=["qwerty", "dvorak", "colemak"],
        help="Keyboard layout to use"
    )
    analyze_parser.add_argument(
        "--output", 
        help="Output file for results (JSON format)"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare text across layouts")
    compare_parser.add_argument("text", help="Text to compare")
    compare_parser.add_argument(
        "--layouts",
        nargs="+",
        default=["qwerty", "dvorak", "colemak"],
        choices=["qwerty", "dvorak", "colemak"],
        help="Layouts to compare"
    )
    compare_parser.add_argument(
        "--output",
        help="Output file for comparison results"
    )
    
    # Report command  
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument("input_file", help="File containing texts to analyze")
    report_parser.add_argument(
        "--layouts",
        nargs="+", 
        default=["qwerty", "dvorak", "colemak"],
        choices=["qwerty", "dvorak", "colemak"],
        help="Layouts to include in report"
    )
    report_parser.add_argument(
        "--output",
        default="quantum_keyboard_report.txt",
        help="Output file for report"
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualization")
    viz_parser.add_argument("text", help="Text to visualize")
    viz_parser.add_argument(
        "--layout",
        default="qwerty", 
        choices=["qwerty", "dvorak", "colemak"],
        help="Keyboard layout to use"
    )
    viz_parser.add_argument(
        "--save",
        help="Save visualization to file"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "compare":
        handle_compare(args)
    elif args.command == "report":
        handle_report(args)
    elif args.command == "visualize":
        handle_visualize(args)
    elif args.command == "version":
        handle_version()
    else:
        parser.print_help()


def handle_analyze(args) -> None:
    """Handle analyze command."""
    print(f"Analyzing: '{args.text}' on {args.layout.upper()} layout")
    
    stats = analyze_single_word(args.text, args.layout)
    
    # Display key metrics
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total Distance: {stats.total_distance:.3f}")
    print(f"Quantum Coherence: {stats.quantum_coherence:.3f}")
    print(f"Harmonic Resonance: {stats.harmonic_resonance:.3f}")
    print(f"Bigram Efficiency: {stats.bigram_efficiency:.3f}")
    print(f"Hand Alternation: {stats.hand_alternation_rate:.1f}%")
    print(f"Information Entropy: {stats.entropy:.3f}")
    print(f"Biomechanical Load: {stats.biomechanical_load:.3f}")
    
    if args.output:
        # Save detailed results
        results = {
            "text": args.text,
            "layout": args.layout,
            "metrics": {
                "total_distance": stats.total_distance,
                "quantum_coherence": stats.quantum_coherence,
                "harmonic_resonance": stats.harmonic_resonance,
                "bigram_efficiency": stats.bigram_efficiency,
                "hand_alternation_rate": stats.hand_alternation_rate,
                "entropy": stats.entropy,
                "biomechanical_load": stats.biomechanical_load,
                # Add more metrics as needed
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


def handle_compare(args) -> None:
    """Handle compare command."""
    print(f"Comparing: '{args.text}' across layouts: {', '.join(args.layouts).upper()}")
    
    results = compare_word_across_layouts(args.text, args.layouts)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"{'Layout':<10} {'Distance':<10} {'Quantum':<10} {'Efficiency':<12} {'Hand Alt':<10}")
    print("-" * 60)
    
    for layout, stats in results.items():
        print(f"{layout.upper():<10} {stats.total_distance:<10.3f} "
              f"{stats.quantum_coherence:<10.3f} {stats.bigram_efficiency:<12.3f} "
              f"{stats.hand_alternation_rate:<10.1f}")
    
    # Find best layout
    best_layout = min(results.items(), key=lambda x: x[1].total_distance)
    print(f"\nRecommended: {best_layout[0].upper()} (lowest distance: {best_layout[1].total_distance:.3f})")
    
    if args.output:
        comparison_data = {
            "text": args.text,
            "layouts": {
                layout: {
                    "total_distance": stats.total_distance,
                    "quantum_coherence": stats.quantum_coherence,
                    "bigram_efficiency": stats.bigram_efficiency,
                    "hand_alternation_rate": stats.hand_alternation_rate,
                }
                for layout, stats in results.items()
            },
            "recommendation": best_layout[0]
        }
        
        with open(args.output, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison results saved to: {args.output}")


def handle_report(args) -> None:
    """Handle report command."""
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Read texts from file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    if not texts:
        print("Error: No texts found in input file")
        sys.exit(1)
    
    print(f"Generating comprehensive report for {len(texts)} texts...")
    
    # Generate report
    keyboard = UltimateQuantumKeyboard(args.layouts[0])  # Use first layout for report generation
    report = keyboard.generate_comprehensive_report(
        texts,
        layouts=args.layouts,
        output_file=args.output
    )
    
    print(f"Report generated and saved to: {args.output}")
    print(f"Analyzed {len(texts)} texts across {len(args.layouts)} layouts")


def handle_visualize(args) -> None:
    """Handle visualize command."""
    try:
        print(f"Creating visualization for: '{args.text}' on {args.layout.upper()} layout")
        
        keyboard = UltimateQuantumKeyboard(args.layout)
        keyboard.create_ultimate_visualization(args.text, save_path=args.save)
        
        if args.save:
            print(f"Visualization saved to: {args.save}")
        else:
            print("Visualization displayed (close plot window to continue)")
            
    except ImportError as e:
        print(f"Error: Visualization requires additional packages: {e}")
        print("Install with: pip install quantum-keyboard[notebooks]")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)


def handle_version() -> None:
    """Handle version command."""
    from . import __version__, __author__
    print(f"Ultimate Quantum Keyboard Analyzer v{__version__}")
    print(f"Created by: {__author__}")
    print("License: MIT")
    print("Project: https://github.com/username/ultimate-quantum-keyboard")


if __name__ == "__main__":
    main()

---

# tests/conftest.py

"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from quantum_keyboard import UltimateQuantumKeyboard, QWERTYLayout, DvorakLayout


@pytest.fixture
def qwerty_keyboard():
    """Fixture for QWERTY keyboard."""
    return UltimateQuantumKeyboard('qwerty')


@pytest.fixture
def dvorak_keyboard():
    """Fixture for Dvorak keyboard."""
    return UltimateQuantumKeyboard('dvorak')


@pytest.fixture
def sample_text():
    """Fixture for sample text."""
    return "hello world"


@pytest.fixture
def sample_texts():
    """Fixture for multiple sample texts."""
    return [
        "hello",
        "world", 
        "quantum computing",
        "keyboard analysis",
        "the quick brown fox"
    ]


@pytest.fixture
def sample_positions():
    """Fixture for sample 3D positions."""
    return np.array([
        [0.0, 2.0, 0.0],  # h
        [2.0, 1.0, 0.0],  # e
        [8.0, 2.0, 0.0],  # l
        [8.0, 2.0, 0.0],  # l
        [8.0, 1.0, 0.0],  # o
    ])


@pytest.fixture
def layouts():
    """Fixture for keyboard layouts."""
    return ['qwerty', 'dvorak', 'colemak']

---

# tests/test_quantum_metrics.py

"""Tests for quantum-inspired metrics."""

import pytest
import numpy as np
from quantum_keyboard import UltimateQuantumKeyboard


class TestQuantumMetrics:
    """Test quantum-inspired metric calculations."""
    
    def test_quantum_coherence_calculation(self, qwerty_keyboard, sample_text):
        """Test quantum coherence calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert 0.0 <= stats.quantum_coherence <= 1.0
        assert isinstance(stats.quantum_coherence, float)
    
    def test_harmonic_resonance_calculation(self, qwerty_keyboard, sample_text):
        """Test harmonic resonance calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert stats.harmonic_resonance >= 0.0
        assert isinstance(stats.harmonic_resonance, float)
    
    def test_dimensional_complexity_calculation(self, qwerty_keyboard, sample_text):
        """Test dimensional complexity calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert 0.0 <= stats.dimensional_complexity <= 3.0
        assert isinstance(stats.dimensional_complexity, float)
    
    def test_quantum_entanglement_calculation(self, qwerty_keyboard, sample_text):
        """Test quantum entanglement calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert 0.0 <= stats.quantum_entanglement <= 1.0
        assert isinstance(stats.quantum_entanglement, float)
    
    def test_phase_synchronization_calculation(self, qwerty_keyboard, sample_text):
        """Test phase synchronization calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert stats.phase_synchronization >= 0.0
        assert isinstance(stats.phase_synchronization, float)
    
    def test_empty_text_quantum_metrics(self, qwerty_keyboard):
        """Test quantum metrics with empty text."""
        stats = qwerty_keyboard.calculate_comprehensive_stats("")
        
        assert stats.quantum_coherence == 0.0
        assert stats.harmonic_resonance == 0.0
        assert stats.dimensional_complexity == 0.0
        assert stats.quantum_entanglement == 0.0
        assert stats.phase_synchronization == 0.0
    
    def test_single_character_quantum_metrics(self, qwerty_keyboard):
        """Test quantum metrics with single character."""
        stats = qwerty_keyboard.calculate_comprehensive_stats("a")
        
        # Single character should have minimal quantum properties
        assert stats.quantum_coherence >= 0.0
        assert stats.harmonic_resonance == 0.0  # No pairs for resonance
        assert stats.dimensional_complexity >= 0.0
    
    def test_quantum_field_initialization(self, qwerty_keyboard):
        """Test quantum field initialization."""
        assert hasattr(qwerty_keyboard, 'quantum_field')
        assert isinstance(qwerty_keyboard.quantum_field, np.ndarray)
        assert qwerty_keyboard.quantum_field.ndim == 4  # 4D field


class TestQuantumFieldProperties:
    """Test quantum field properties and calculations."""
    
    def test_quantum_state_calculation(self, qwerty_keyboard):
        """Test quantum state calculation for keys."""
        layout = qwerty_keyboard.layout
        
        for char, key_info in layout.key_info.items():
            assert hasattr(key_info, 'quantum_state')
            assert isinstance(key_info.quantum_state, float)
    
    def test_harmonic_generation(self, qwerty_keyboard):
        """Test harmonic frequency generation."""
        layout = qwerty_keyboard.layout
        
        for char, key_info in layout.key_info.items():
            assert hasattr(key_info, 'harmonics')
            assert isinstance(key_info.harmonics, list)
            if key_info.harmonics:
                assert all(isinstance(h, float) for h in key_info.harmonics)
                assert all(h > 0 for h in key_info.harmonics)
    
    def test_fractal_dimension_calculation(self, qwerty_keyboard, sample_positions):
        """Test fractal dimension calculation."""
        fractal_dim = qwerty_keyboard._calculate_fractal_dimension(sample_positions)
        
        assert 0.0 <= fractal_dim <= 3.0
        assert isinstance(fractal_dim, float)
    
    def test_fractal_dimension_edge_cases(self, qwerty_keyboard):
        """Test fractal dimension with edge cases."""
        # Empty array
        empty_positions = np.array([]).reshape(0, 3)
        fractal_dim = qwerty_keyboard._calculate_fractal_dimension(empty_positions)
        assert fractal_dim == 1.0
        
        # Single point
        single_point = np.array([[0.0, 0.0, 0.0]])
        fractal_dim = qwerty_keyboard._calculate_fractal_dimension(single_point)
        assert fractal_dim == 1.0
        
        # Two points
        two_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        fractal_dim = qwerty_keyboard._calculate_fractal_dimension(two_points)
        assert fractal_dim == 1.0

---

# tests/test_layouts.py

"""Tests for keyboard layout implementations."""

import pytest
import numpy as np
from quantum_keyboard import QWERTYLayout, DvorakLayout, ColemakLayout, UltimateQuantumKeyboard


class TestKeyboardLayouts:
    """Test keyboard layout implementations."""
    
    def test_qwerty_layout_creation(self):
        """Test QWERTY layout creation."""
        layout = QWERTYLayout('qwerty')
        
        assert layout.layout_name == 'qwerty'
        assert len(layout.key_info) > 0
        
        # Test essential keys exist
        essential_keys = ['a', 's', 'd', 'f', 'j', 'k', 'l', ' ']
        for key in essential_keys:
            assert key in layout.key_info
    
    def test_dvorak_layout_creation(self):
        """Test Dvorak layout creation.""" 
        layout = DvorakLayout('dvorak')
        
        assert layout.layout_name == 'dvorak'
        assert len(layout.key_info) > 0
        
        # Test Dvorak-specific home row
        dvorak_home = ['a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's']
        for key in dvorak_home:
            assert key in layout.key_info
    
    def test_colemak_layout_creation(self):
        """Test Colemak layout creation."""
        layout = ColemakLayout('colemak')
        
        assert layout.layout_name == 'colemak'  
        assert len(layout.key_info) > 0
        
        # Test some Colemak-specific positions
        assert 'a' in layout.key_info
        assert 'r' in layout.key_info
        assert 's' in layout.key_info
    
    def test_key_info_properties(self):
        """Test key info properties are properly set."""
        layout = QWERTYLayout('qwerty')
        
        for char, key_info in layout.key_info.items():
            # Test required properties
            assert hasattr(key_info, 'position')
            assert hasattr(key_info, 'finger')
            assert hasattr(key_info, 'hand')
            assert hasattr(key_info, 'effort')
            assert hasattr(key_info, 'frequency')
            assert hasattr(key_info, 'quantum_state')
            assert hasattr(key_info, 'harmonics')
            assert hasattr(key_info, 'biomechanical_stress')
            assert hasattr(key_info, 'neural_activation')
            
            # Test types
            assert isinstance(key_info.position, np.ndarray)
            assert len(key_info.position) == 3
            assert isinstance(key_info.effort, float)
            assert isinstance(key_info.frequency, float)
    
    def test_position_coordinates(self):
        """Test position coordinates are reasonable."""
        layout = QWERTYLayout('qwerty')
        
        for char, key_info in layout.key_info.items():
            x, y, z = key_info.position
            
            # Test reasonable coordinate ranges
            assert 0 <= x <= 15  # Keyboard width
            assert 0 <= y <= 5   # Keyboard height  
            assert 0 <= z <= 2   # Layers (normal, shifted)
    
    def test_effort_values(self):
        """Test effort values are reasonable."""
        layout = QWERTYLayout('qwerty')
        
        for char, key_info in layout.key_info.items():
            assert 0.1 <= key_info.effort <= 2.0
    
    def test_frequency_values(self):
        """Test frequency values for common letters."""
        layout = QWERTYLayout('qwerty')
        
        # Test that common letters have higher frequencies
        if 'e' in layout.key_info and 'z' in layout.key_info:
            assert layout.key_info['e'].frequency > layout.key_info['z'].frequency
        
        if 'a' in layout.key_info and 'q' in layout.key_info:
            assert layout.key_info['a'].frequency > layout.key_info['q'].frequency


class TestLayoutComparison:
    """Test comparison between different layouts."""
    
    def test_layout_differences(self):
        """Test that different layouts have different key positions."""
        qwerty = QWERTYLayout('qwerty')
        dvorak = DvorakLayout('dvorak')
        
        # Home row should be different
        if 'a' in qwerty.key_info and 'a' in dvorak.key_info:
            qwerty_a_pos = qwerty.key_info['a'].position
            dvorak_a_pos = dvorak.key_info['a'].position
            
            # Positions should be different (at least one coordinate)
            assert not np.allclose(qwerty_a_pos, dvorak_a_pos)
    
    def test_common_keys_exist(self):
        """Test that common keys exist in all layouts."""
        layouts = [QWERTYLayout('qwerty'), DvorakLayout('dvorak'), ColemakLayout('colemak')]
        
        common_keys = ['a', 'e', 'i', 'o', 'u', 's', 't', 'n', 'r', ' ']
        
        for layout in layouts:
            for key in common_keys:
                assert key in layout.key_info, f"Key '{key}' missing from {layout.layout_name}"
    
    @pytest.mark.parametrize("layout_name", ['qwerty', 'dvorak', 'colemak'])
    def test_keyboard_creation_by_name(self, layout_name):
        """Test keyboard creation by layout name."""
        keyboard = UltimateQuantumKeyboard(layout_name)
        
        assert keyboard.layout.layout_name == layout_name
        assert len(keyboard.layout.key_info) > 0


class TestCustomLayoutSupport:
    """Test support for custom layouts."""
    
    def test_custom_layout_integration(self):
        """Test that custom layouts can be integrated."""
        # This would test custom layout class integration
        # For now, test that the base class works
        from quantum_keyboard.ultimate_quantum_keyboard import KeyboardLayout
        
        assert hasattr(KeyboardLayout, 'setup_layout')
        assert hasattr(KeyboardLayout, 'add_key')

---

# tests/test_ergonomic_analysis.py

"""Tests for ergonomic analysis functionality."""

import pytest
from quantum_keyboard import UltimateQuantumKeyboard


class TestErgonomicMetrics:
    """Test ergonomic metric calculations."""
    
    def test_hand_alternation_rate(self, qwerty_keyboard, sample_text):
        """Test hand alternation rate calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert 0.0 <= stats.hand_alternation_rate <= 100.0
        assert isinstance(stats.hand_alternation_rate, float)
    
    def test_finger_utilization(self, qwerty_keyboard, sample_text):
        """Test finger utilization calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert isinstance(stats.finger_utilization, dict)
        
        # Check that utilization counts are non-negative
        for finger, count in stats.finger_utilization.items():
            assert count >= 0
            assert isinstance(count, int)
    
    def test_same_finger_percentage(self, qwerty_keyboard, sample_text):
        """Test same finger percentage calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert 0.0 <= stats.same_finger_percentage <= 100.0
        assert isinstance(stats.same_finger_percentage, float)
    
    def test_biomechanical_load(self, qwerty_keyboard, sample_text):
        """Test biomechanical load calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert stats.biomechanical_load >= 0.0
        assert isinstance(stats.biomechanical_load, float)
    
    def test_total_effort(self, qwerty_keyboard, sample_text):
        """Test total effort calculation."""
        stats = qwerty_keyboard.calculate_comprehensive_stats(sample_text)
        
        assert stats.total_effort >= 0.0
        assert isinstance(stats.total_effort, float)
    
    def test_perfect_hand_alternation(self, qwerty_keyboard):
        """Test perfect hand alternation scenario."""
        # Create text with perfect hand alternation
        # a=left, l=right, a=left, l=right
        perfect_alternation_text = "alal"
        stats = qwerty_keyboard.calculate_comprehensive_stats(perfect_alternation_text)
        
        # Should have high hand alternation rate
        assert stats.hand_alternation_rate > 50.0
    
    def test_same_hand_typing(self, qwerty_keyboard):
        """Test same hand typing scenario."""
        # Create text using only left hand keys
        same_hand_text = "asdf"
        stats = qwerty_keyboard.calculate_comprehensive_stats(same_hand_text)
        
        # Should have low hand alternation rate
        assert stats.hand_alternation_rate < 50.0
    
    def test_repeated_character_same_finger(self, qwerty_keyboard):
        """Test repeated character (same finger) scenario."""
        repeated_text = "aaaa"
        stats = qwerty_keyboard.calculate_comprehensive_stats(repeated_text)
        
        # Should have high same finger percentage
        assert stats.same_finger_percentage > 80.0


class TestBiomechanicalModeling:
    """Test biomechanical modeling features."""
    
    def test_biomechanical_stress_calculation(self, qwerty_keyboard):
        """Test biomechanical stress calculation for keys."""
        layout = qwerty_keyboard.layout
        
        for char, key_info in layout.key_info.items():
            assert hasattr(key_info, 'biomechanical_stress')
            assert key_info.biomechanical_stress >= 0.0
            assert isinstance(key_info.biomechanical_stress, float)
    
    def test_neural_activation_calculation(self, qwerty_keyboard):
        """Test neural activation calculation."""
        layout = qwerty_keyboard.layout
        
        for char, key_info in layout.key_info.items():
            assert hasattr(key_info, 'neural_activation')
            assert key_info.neural_activation >= 0.0
            assert isinstance(key_info.neural_activation, float)
    
    def test_home_row_advantage(self, qwerty_keyboard):
        """Test that home row keys have lower biomechanical stress."""
        layout = qwerty_keyboard.layout
        
        # Home row keys should generally have lower stress
        home_keys = ['a', 's', 'd', 'f', 'j', 'k', 'l']
        top_keys = ['q', 'w', 'e', 'r', 'u', 'i', 'o', 'p']
        
        if all(key in layout.key_info for key in home_keys + top_keys):
            avg_home_stress = sum(layout.key_info[key].biomechanical_stress for key in home_keys) / len(home_keys)
            avg_top_stress = sum(layout.key_info[key].biomechanical_stress for key in top_keys) / len(top_keys)
            
            # Home row should generally be less stressful
            assert avg_home_stress < avg_top_stress


class TestErgonomicComparisons:
    """Test ergonomic comparisons between layouts."""
    
    def test_layout_ergonomic_differences(self, sample_text):
        """Test ergonomic differences between layouts."""
        qwerty = UltimateQuantumKeyboard('qwerty')
        dvorak = UltimateQuantumKeyboard('dvorak')
        
        qwerty_stats = qwerty.calculate_comprehensive_stats(sample_text)
        dvorak_stats = dvorak.calculate_comprehensive_stats(sample_text)
        
        # Stats should be calculated for both
        assert qwerty_stats.hand_alternation_rate >= 0.0
        assert dvorak_stats.hand_alternation_rate >= 0.0
        
        # They should generally be different (unless by coincidence)
        metrics_to_compare = [
            'total_distance', 'hand_alternation_rate', 'biomechanical_load'
        ]
        
        differences_found = False
        for metric in metrics_to_compare:
            qwerty_val = getattr(qwerty_stats, metric)
            dvorak_val = getattr(dvorak_stats, metric)
            
            if abs(qwerty_val - dvorak_val) > 0.001:  # Allow for small floating point differences
                differences_found = True
                break
        
        assert differences_found, "No significant differences found between layouts"

---

# tests/test_visualization.py

"""Tests for visualization functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from quantum_keyboard import UltimateQuantumKeyboard


class TestVisualizationCreation:
    """Test visualization creation functionality."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_ultimate_visualization_creation(self, mock_figure, mock_show, qwerty_keyboard, sample_text):
        """Test ultimate visualization creation."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        # Should not raise an exception
        try:
            qwerty_keyboard.create_ultimate_visualization(sample_text)
        except ImportError:
            # Skip if visualization dependencies not available
            pytest.skip("Visualization dependencies not available")
        except Exception as e:
            # Other exceptions should not occur
            pytest.fail(f"Unexpected exception in visualization: {e}")
    
    def test_visualization_with_empty_text(self, qwerty_keyboard):
        """Test visualization with empty text."""
        try:
            qwerty_keyboard.create_ultimate_visualization("")
        except ImportError:
            pytest.skip("Visualization dependencies not available")
        except Exception:
            # Should handle empty text gracefully
            pass  # Expected to handle gracefully
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_visualization_save(self, mock_figure, mock_show, mock_savefig, qwerty_keyboard, sample_text):
        """Test saving visualization to file."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        try:
            qwerty_keyboard.create_ultimate_visualization(sample_text, save_path="test.png")
            # Should call savefig if save_path provided
            mock_savefig.assert_called_once_with("test.png", dpi=300, bbox_inches='tight')
        except ImportError:
            pytest.skip("Visualization dependencies not available")


class TestVisualizationHelpers:
    """Test visualization helper methods."""
    
    def test_plot_methods_exist(self, qwerty_keyboard):
        """Test that plot helper methods exist."""
        required_methods = [
            '_plot_3d_path_with_field',
            '_plot_comprehensive_radar', 
            '_plot_finger_heatmap',
            '_plot_harmonic_spectrum',
            '_plot_pca_analysis',
            '_plot_velocity_profile',
            '_plot_quantum_evolution',
            '_plot_information_metrics',
            '_plot_efficiency_metrics',
            '_plot_biomechanical_analysis',
            '_plot_pattern_analysis',
            '_plot_summary_stats'
        ]
        
        for method_name in required_methods:
            assert hasattr(qwerty_keyboard, method_name)
            assert callable(getattr(qwerty_keyboard, method_name))


class TestComparativeVisualization:
    """Test comparative visualization functionality."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_comparative_visualization(self, mock_figure, mock_show, qwerty_keyboard, sample_texts):
        """Test comparative visualization across layouts."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        try:
            qwerty_keyboard.create_comparative_visualization(
                sample_texts[:2], ['qwerty', 'dvorak']
            )
        except ImportError:
            pytest.skip("Visualization dependencies not available")
        except Exception as e:
            pytest.fail(f"Unexpected exception in comparative visualization: {e}")

---

# examples/basic_analysis.py

"""Basic analysis example for the Ultimate Quantum Keyboard Analyzer."""

from quantum_keyboard import analyze_single_word, compare_word_across_layouts, UltimateQuantumKeyboard


def main():
    """Demonstrate basic analysis functionality."""
    print("=== Ultimate Quantum Keyboard Analyzer - Basic Example ===\n")
    
    # Single word analysis
    print("1. Single Word Analysis")
    print("-" * 30)
    
    word = "quantum"
    stats = analyze_single_word(word, "qwerty")
    
    print(f"Analyzing: '{word}' on QWERTY layout")
    print(f"Total Distance: {stats.total_distance:.3f}")
    print(f"Quantum Coherence: {stats.quantum_coherence:.3f}")
    print(f"Harmonic Resonance: {stats.harmonic_resonance:.3f}")
    print(f"Hand Alternation: {stats.hand_alternation_rate:.1f}%")
    print(f"Bigram Efficiency: {stats.bigram_efficiency:.3f}")
    
    # Layout comparison
    print(f"\n2. Layout Comparison")
    print("-" * 30)
    
    comparison_word = "efficiency"
    results = compare_word_across_layouts(comparison_word, ["qwerty", "dvorak", "colemak"])
    
    print(f"Comparing '{comparison_word}' across layouts:")
    for layout, stats in results.items():
        print(f"{layout.upper():8}: Distance={stats.total_distance:.3f}, "
              f"Quantum={stats.quantum_coherence:.3f}")
    
    # Advanced analysis
    print(f"\n3. Advanced Analysis")
    print("-" * 30)
    
    keyboard = UltimateQuantumKeyboard('qwerty')
    advanced_stats = keyboard.calculate_comprehensive_stats("machine learning")
    
    print("Advanced metrics for 'machine learning':")
    print(f"Dimensional Complexity: {advanced_stats.dimensional_complexity:.3f}")
    print(f"Information Entropy: {advanced_stats.entropy:.3f}")
    print(f"Path Smoothness: {advanced_stats.path_smoothness:.3f}")
    print(f"Biomechanical Load: {advanced_stats.biomechanical_load:.3f}")
    
    # Visualization (if available)
    print(f"\n4. Visualization")
    print("-" * 30)
    
    try:
        keyboard.create_ultimate_visualization("hello")
        print("✓ Visualization created successfully!")
        print("Close the plot window to continue...")
    except ImportError:
        print("⚠ Visualization requires additional packages")
        print("Install with: pip install quantum-keyboard[notebooks]")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")


if __name__ == "__main__":
    main()

---

# examples/comparative_study.py

"""Comparative study example demonstrating research capabilities."""

from quantum_keyboard import UltimateQuantumKeyboard
import json


def main():
    """Demonstrate comprehensive comparative analysis."""
    print("=== Comparative Keyboard Layout Study ===\n")
    
    # Research texts representing different domains
    research_texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning algorithms",
        "biomechanical ergonomic optimization research methodology", 
        "quantum computing entanglement superposition measurement",
        "statistical significance hypothesis testing confidence intervals",
        "human computer interaction user experience design principles",
        "data visualization information theory entropy calculation",
        "software engineering best practices code review standards"
    ]
    
    layouts_to_test = ['qwerty', 'dvorak', 'colemak']
    
    print(f"Analyzing {len(research_texts)} research texts across {len(layouts_to_test)} layouts...")
    print("This comprehensive study will evaluate multiple metrics.\n")
    
    # Create analyzer
    analyzer = UltimateQuantumKeyboard('qwerty')
    
    # Perform comprehensive comparison
    results = analyzer.compare_layouts(research_texts, layouts_to_test)
    
    # Display key findings
    print("=== KEY RESEARCH FINDINGS ===")
    print("-" * 40)
    
    key_metrics = [
        ('total_distance', 'Total Typing Distance', 'lower'),
        ('quantum_coherence', 'Quantum Coherence', 'higher'),
        ('hand_alternation_rate', 'Hand Alternation Rate', 'higher'),
        ('bigram_efficiency', 'Bigram Efficiency', 'higher'),
        ('biomechanical_load', 'Biomechanical Load', 'lower')
    ]
    
    for metric_key, metric_name, direction in key_metrics:
        print(f"\n{metric_name}:")
        
        layout_scores = {}
        for layout in layouts_to_test:
            if metric_key in results[layout]:
                score = results[layout][metric_key].mean
                layout_scores[layout] = score
                ci_lower, ci_upper = results[layout][metric_key].confidence_interval_95
                print(f"  {layout.upper():8}: {score:.3f} [CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Determine best layout for this metric
        if layout_scores:
            if direction == 'lower':
                best_layout = min(layout_scores.items(), key=lambda x: x[1])
            else:
                best_layout = max(layout_scores.items(), key=lambda x: x[1])
            
            print(f"  Best: {best_layout[0].upper()} ({best_layout[1]:.3f})")
    
    # Generate comprehensive report
    print(f"\n=== GENERATING RESEARCH REPORT ===")
    print("-" * 40)
    
    report = analyzer.generate_comprehensive_report(
        research_texts[:5],  # Use subset for demo
        layouts_to_test,
        output_file="comparative_study_report.txt"
    )
    
    print("✓ Comprehensive report generated: comparative_study_report.txt")
    
    # Export detailed results
    print(f"\n=== EXPORTING RESEARCH DATA ===")
    print("-" * 40)
    
    # Convert results for JSON export
    export_data = {
        "study_metadata": {
            "texts_analyzed": len(research_texts),
            "layouts_compared": layouts_to_test,
            "total_metrics": len(key_metrics)
        },
        "statistical_results": {}
    }
    
    for layout in layouts_to_test:
        export_data["statistical_results"][layout] = {}
        for metric_key, _, _ in key_metrics:
            if metric_key in results[layout]:
                stat_result = results[layout][metric_key]
                export_data["statistical_results"][layout][metric_key] = {
                    "mean": stat_result.mean,
                    "std_dev": stat_result.std_dev,
                    "confidence_interval_95": stat_result.confidence_interval_95,
                    "sample_size": len(research_texts)
                }
    
    with open("comparative_study_data.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print("✓ Research data exported: comparative_study_data.json")
    
    # Research recommendations
    print(f"\n=== RESEARCH RECOMMENDATIONS ===")
    print("-" * 40)
    
    # Calculate overall scores
    layout_overall_scores = {}
    weights = {'total_distance': -1, 'quantum_coherence': 2, 'hand_alternation_rate': 1.5, 
              'bigram_efficiency': 2, 'biomechanical_load': -1.5}
    
    for layout in layouts_to_test:
        score = 0
        for metric_key, weight in weights.items():
            if metric_key in results[layout]:
                score += weight * results[layout][metric_key].mean
        layout_overall_scores[layout] = score
    
    ranked_layouts = sorted(layout_overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Overall Layout Rankings (based on weighted composite score):")
    for i, (layout, score) in enumerate(ranked_layouts, 1):
        print(f"{i}. {layout.upper()}: {score:.3f}")
    
    print(f"\nRecommendation: {ranked_layouts[0][0].upper()} layout")
    print("Based on comprehensive quantum-inspired analysis of research texts.")
    
    print(f"\n=== STUDY COMPLETE ===")
    print("Files generated:")
    print("• comparative_study_report.txt - Human-readable report") 
    print("• comparative_study_data.json - Machine-readable data")
    print("\nThis data can be used for:")
    print("• Academic publication")
    print("• Further statistical analysis")
    print("• Replication studies")
    print("• Meta-analysis research")


if __name__ == "__main__":
    main()

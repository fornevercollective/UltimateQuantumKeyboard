return {
    'curvature': curvature,
    'torsion': torsion,
    'planarity': planarity,
    'compactness': compactness,
    'path_smoothness': float(smoothness)
}


def _calculate_quantum_metrics(self, positions: np.ndarray, text: str) -> Dict[str, float]:
    """Calculate quantum-inspired metrics."""
    if len(positions) < 2:
        return {
            'quantum_coherence': 0.0,
            'harmonic_resonance': 0.0,
            'dimensional_complexity': 0.0,
            'quantum_entanglement': 0.0,
            'phase_synchronization': 0.0
        }

    # Quantum coherence (phase stability)
    phases = []
    for i in range(len(positions) - 1):
        pos1, pos2 = positions[i], positions[i + 1]
        if len(pos1) >= 2 and len(pos2) >= 2:
            phase_diff = np.arctan2(np.cross(pos1[:2], pos2[:2]), np.dot(pos1[:2], pos2[:2]))
            phases.append(phase_diff)

    if phases:
        coherence = 1.0 / (1.0 + np.std(phases))
    else:
        coherence = 1.0

    # Harmonic resonance
    total_resonance = 0.0
    valid_pairs = 0

    for i in range(len(text) - 1):
        char1, char2 = text[i].lower(), text[i + 1].lower()
        if char1 in self.layout.key_info and char2 in self.layout.key_info:
            harmonics1 = self.layout.key_info[char1].harmonics
            harmonics2 = self.layout.key_info[char2].harmonics

            if harmonics1 and harmonics2:
                resonance = 0.0
                for h1 in harmonics1[:3]:
                    for h2 in harmonics2[:3]:
                        if h2 != 0:
                            ratio = h1 / h2
                            if 0.5 <= ratio <= 2.0:
                                resonance += 1.0 / abs(ratio - 1.0 + 0.1)

                total_resonance += resonance
                valid_pairs += 1

    harmonic_resonance = total_resonance / max(1, valid_pairs)

    # Dimensional complexity (fractal dimension estimation)
    dimensional_complexity = self._calculate_fractal_dimension(positions)

    # Quantum entanglement (correlation between distant keys)
    entanglement = self._calculate_quantum_entanglement(positions)

    # Phase synchronization
    phase_sync = self._calculate_phase_synchronization(positions)

    return {
        'quantum_coherence': float(coherence),
        'harmonic_resonance': float(harmonic_resonance),
        'dimensional_complexity': float(dimensional_complexity),
        'quantum_entanglement': float(entanglement),
        'phase_synchronization': float(phase_sync)
    }


def _calculate_information_metrics(self, text: str, positions: np.ndarray) -> Dict[str, float]:
    """Calculate information theory metrics."""
    # Shannon entropy
    char_counts = Counter(text.lower())
    total_chars = len(text)

    if total_chars == 0:
        return {
            'entropy': 0.0,
            'mutual_information': 0.0,
            'kolmogorov_complexity': 0.0
        }

    probabilities = [count / total_chars for count in char_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    # Mutual information between consecutive characters
    bigram_counts = defaultdict(int)
    for i in range(len(text) - 1):
        bigram = text[i:i + 2].lower()
        bigram_counts[bigram] += 1

    total_bigrams = sum(bigram_counts.values())
    if total_bigrams > 0:
        mi = 0.0
        for bigram, count in bigram_counts.items():
            if len(bigram) == 2:
                p_xy = count / total_bigrams
                p_x = char_counts.get(bigram[0], 0) / total_chars
                p_y = char_counts.get(bigram[1], 0) / total_chars

                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))

        mutual_information = mi
    else:
        mutual_information = 0.0

    # Kolmogorov complexity approximation (compression ratio)
    if len(positions) > 0:
        try:
            compressed = zlib.compress(json.dumps(positions.tolist()).encode())
            original_size = len(json.dumps(positions.tolist()).encode())
            kolmogorov_complexity = len(compressed) / max(1, original_size)
        except:
            kolmogorov_complexity = 1.0
    else:
        kolmogorov_complexity = 0.0

    return {
        'entropy': float(entropy),
        'mutual_information': float(mutual_information),
        'kolmogorov_complexity': float(kolmogorov_complexity)
    }


def _calculate_pattern_metrics(self, text: str) -> Dict[str, float]:
    """Calculate pattern analysis metrics."""
    if len(text) < 2:
        return {
            'bigram_efficiency': 0.0,
            'trigram_efficiency': 0.0,
            'ngram_entropy': 0.0,
            'rhythm_consistency': 0.0
        }

    # Bigram efficiency
    bigram_efficiency = self._calculate_bigram_efficiency(text)

    # Trigram efficiency
    trigram_efficiency = self._calculate_trigram_efficiency(text)

    # N-gram entropy
    ngram_entropy = self._calculate_ngram_entropy(text)

    # Rhythm consistency (based on timing patterns)
    rhythm_consistency = self._calculate_rhythm_consistency(text)

    return {
        'bigram_efficiency': bigram_efficiency,
        'trigram_efficiency': trigram_efficiency,
        'ngram_entropy': ngram_entropy,
        'rhythm_consistency': rhythm_consistency
    }


def _calculate_ml_metrics(self, positions: np.ndarray, text: str) -> Dict[str, float]:
    """Calculate machine learning derived metrics."""
    if len(positions) < 3:
        return {
            'pca_variance_explained': 0.0,
            'clustering_quality': 0.0,
            'anomaly_score': 0.0
        }

    # PCA analysis
    try:
        pca = PCA()
        pca.fit(positions)
        pca_variance = float(sum(pca.explained_variance_ratio_[:2]))
    except:
        pca_variance = 0.0

    # Clustering quality
    try:
        if len(positions) >= 4:
            kmeans = KMeans(n_clusters=min(3, len(positions)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(positions)
            if len(set(cluster_labels)) > 1:
                clustering_quality = silhouette_score(positions, cluster_labels)
            else:
                clustering_quality = 0.0
        else:
            clustering_quality = 0.0
    except:
        clustering_quality = 0.0

    # Anomaly score (distance from centroid)
    try:
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        anomaly_score = float(np.mean(distances) / (np.std(distances) + 1e-8))
    except:
        anomaly_score = 0.0

    return {
        'pca_variance_explained': pca_variance,
        'clustering_quality': float(clustering_quality),
        'anomaly_score': anomaly_score
    }


def _calculate_temporal_metrics(self, positions: np.ndarray) -> Dict[str, Union[float, List[float]]]:
    """Calculate temporal and velocity metrics."""
    if len(positions) < 2:
        return {
            'typing_rhythm': 0.0,
            'acceleration_variance': 0.0,
            'velocity_profile': []
        }

    # Velocity profile (assuming unit time steps)
    velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=1)

    # Typing rhythm (consistency of velocity)
    if len(velocities) > 1:
        rhythm = 1.0 / (1.0 + np.std(velocities))
    else:
        rhythm = 1.0

    # Acceleration variance
    if len(velocities) > 1:
        accelerations = velocities[1:] - velocities[:-1]
        acc_variance = float(np.var(accelerations)) if len(accelerations) > 0 else 0.0
    else:
        acc_variance = 0.0

    return {
        'typing_rhythm': float(rhythm),
        'acceleration_variance': acc_variance,
        'velocity_profile': velocities.tolist()
    }


# ========================== HELPER CALCULATION METHODS ==========================

def _calculate_curvature(self, positions: np.ndarray) -> float:
    """Calculate path curvature."""
    if len(positions) < 3:
        return 0.0

    vectors = positions[1:] - positions[:-1]
    vector_norms = np.linalg.norm(vectors, axis=1)

    valid_indices = vector_norms > 1e-8
    if np.sum(valid_indices) < 2:
        return 0.0

    vectors = vectors[valid_indices]
    vector_norms = vector_norms[valid_indices]

    if len(vectors) < 2:
        return 0.0

    cross_products = np.cross(vectors[:-1], vectors[1:])
    cross_norms = np.linalg.norm(cross_products, axis=1)
    norm_products = vector_norms[:-1] * vector_norms[1:]

    valid_curvatures = norm_products > 1e-8
    if not np.any(valid_curvatures):
        return 0.0

    curvatures = cross_norms[valid_curvatures] / norm_products[valid_curvatures]
    return float(np.mean(curvatures))


def _calculate_torsion(self, positions: np.ndarray) -> float:
    """Calculate path torsion."""
    if len(positions) < 4:
        return 0.0

    v1 = positions[1:-2] - positions[0:-3]
    v2 = positions[2:-1] - positions[1:-2]
    v3 = positions[3:] - positions[2:-1]

    cross1 = np.cross(v1, v2)
    cross2 = np.cross(v2, v3)

    cross1_norms = np.linalg.norm(cross1, axis=1)
    cross2_norms = np.linalg.norm(cross2, axis=1)

    valid_indices = (cross1_norms > 1e-8) & (cross2_norms > 1e-8)
    if not np.any(valid_indices):
        return 0.0

    valid_cross1 = cross1[valid_indices]
    valid_cross2 = cross2[valid_indices]
    valid_cross1_norms = cross1_norms[valid_indices]
    valid_cross2_norms = cross2_norms[valid_indices]

    dot_products = np.sum(valid_cross1 * valid_cross2, axis=1)
    cos_torsions = np.clip(dot_products / (valid_cross1_norms * valid_cross2_norms), -1.0, 1.0)

    return float(np.mean(cos_torsions))


def _calculate_compactness(self, positions: np.ndarray) -> float:
    """Calculate path compactness."""
    if len(positions) < 2:
        return 0.0

    consecutive_distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    avg_consecutive = np.mean(consecutive_distances)

    try:
        from scipy.spatial.distance import pdist
        max_distance = np.max(pdist(positions))
    except ImportError:
        ranges = np.ptp(positions, axis=0)
        max_distance = np.linalg.norm(ranges)

    if max_distance == 0:
        return 0.0

    return float(avg_consecutive / max_distance)


def _calculate_fractal_dimension(self, positions: np.ndarray) -> float:
    """Estimate fractal dimension using box-counting method."""
    if len(positions) < 3:
        return 1.0

    def box_count(points, epsilon):
        # Normalize points to unit cube
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        normalized = (points - min_vals) / ranges

        # Count boxes
        boxes = set()
        for point in normalized:
            box = tuple((point / epsilon).astype(int))
            boxes.add(box)

        return len(boxes)

    # Use multiple scales
    epsilons = np.logspace(-2, 0, 10)
    counts = [box_count(positions, eps) for eps in epsilons]

    # Fit line in log-log space
    log_eps = np.log(epsilons)
    log_counts = np.log(counts)

    # Remove invalid values
    valid = (log_counts > 0) & np.isfinite(log_counts)
    if np.sum(valid) < 2:
        return 1.0

    try:
        slope = np.polyfit(log_eps[valid], log_counts[valid], 1)[0]
        return float(np.clip(-slope, 0, 3))
    except:
        return 1.0


def _calculate_quantum_entanglement(self, positions: np.ndarray) -> float:
    """Calculate quantum entanglement between distant positions."""
    if len(positions) < 4:
        return 0.0

    # Calculate correlations between non-adjacent positions
    entanglement = 0.0
    count = 0

    for i in range(len(positions)):
        for j in range(i + 2, len(positions)):
            if j < len(positions):
                # Calculate correlation between positions
                pos1, pos2 = positions[i], positions[j]
                distance = np.linalg.norm(pos1 - pos2)
                correlation = np.exp(-distance / 5.0)  # Exponential decay
                entanglement += correlation
                count += 1

    return entanglement / max(1, count)


def _calculate_phase_synchronization(self, positions: np.ndarray) -> float:
    """Calculate phase synchronization across the path."""
    if len(positions) < 3:
        return 0.0

    # Calculate instantaneous phases
    phases = []
    for i in range(len(positions) - 1):
        dx = positions[i + 1] - positions[i]
        phase = np.arctan2(dx[1], dx[0]) if len(dx) >= 2 else 0.0
        phases.append(phase)

    if len(phases) < 2:
        return 0.0

    # Calculate phase synchronization index
    phases = np.array(phases)
    phase_diffs = np.diff(phases)

    # Wrap phase differences to [-π, π]
    phase_diffs = np.arctan2(np.sin(phase_diffs), np.cos(phase_diffs))

    # Synchronization is inverse of phase variance
    sync_index = 1.0 / (1.0 + np.var(phase_diffs))
    return float(sync_index)


def _calculate_bigram_efficiency(self, text: str) -> float:
    """Calculate bigram typing efficiency."""
    if len(text) < 2:
        return 0.0

    total_efficiency = 0.0
    valid_bigrams = 0

    for i in range(len(text) - 1):
        char1, char2 = text[i].lower(), text[i + 1].lower()

        if char1 in self.layout.key_info and char2 in self.layout.key_info:
            key1, key2 = self.layout.key_info[char1], self.layout.key_info[char2]

            efficiency = 1.0

            # Hand alternation bonus
            if key1.hand != key2.hand:
                efficiency *= 1.3

            # Same finger penalty
            if key1.finger == key2.finger:
                efficiency *= 0.2

            # Distance penalty
            distance = np.linalg.norm(key1.position - key2.position)
            efficiency *= max(0.1, 1.0 - distance / 10.0)

            # Effort consideration
            avg_effort = (key1.effort + key2.effort) / 2.0
            efficiency *= max(0.1, 2.0 - avg_effort)

            total_efficiency += efficiency
            valid_bigrams += 1

    return total_efficiency / max(1, valid_bigrams)


def _calculate_trigram_efficiency(self, text: str) -> float:
    """Calculate trigram typing efficiency."""
    if len(text) < 3:
        return 0.0

    total_efficiency = 0.0
    valid_trigrams = 0

    for i in range(len(text) - 2):
        chars = [text[i + j].lower() for j in range(3)]

        if all(c in self.layout.key_info for c in chars):
            keys = [self.layout.key_info[c] for c in chars]

            efficiency = 1.0

            # Hand alternation pattern
            hands = [k.hand for k in keys]
            if hands[0] != hands[1] and hands[1] != hands[2]:
                efficiency *= 1.4  # Perfect alternation
            elif hands[0] == hands[1] == hands[2]:
                efficiency *= 0.6  # Same hand

            # Finger diversity
            fingers = [k.finger for k in keys]
            unique_fingers = len(set(fingers))
            if unique_fingers == 1:
                efficiency *= 0.1  # Very bad
            elif unique_fingers == 2:
                efficiency *= 0.5
            else:
                efficiency *= 1.2  # Good diversity

            # Total path distance
            dist1 = np.linalg.norm(keys[1].position - keys[0].position)
            dist2 = np.linalg.norm(keys[2].position - keys[1].position)
            total_dist = dist1 + dist2
            efficiency *= max(0.1, 1.0 - total_dist / 15.0)

            total_efficiency += efficiency
            valid_trigrams += 1

    return total_efficiency / max(1, valid_trigrams)


def _calculate_ngram_entropy(self, text: str) -> float:
    """Calculate n-gram entropy for pattern complexity."""
    if len(text) < 2:
        return 0.0

    # Calculate entropy for different n-gram sizes
    entropies = []

    for n in range(2, min(6, len(text) + 1)):
        ngrams = [text[i:i + n].lower() for i in range(len(text) - n + 1)]
        ngram_counts = Counter(ngrams)
        total = len(ngrams)

        if total > 0:
            probs = [count / total for count in ngram_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def _calculate_rhythm_consistency(self, text: str) -> float:
    """Calculate typing rhythm consistency."""
    if len(text) < 3:
        return 1.0

    # Simulate typing intervals based on key difficulty
    intervals = []
    for char in text.lower():
        if char in self.layout.key_info:
            # Base interval modified by key effort
            effort = self.layout.key_info[char].effort
            interval = 0.1 + effort * 0.05  # Base time + effort penalty
            intervals.append(interval)

    if len(intervals) < 2:
        return 1.0

    # Rhythm consistency is inverse of interval variance
    consistency = 1.0 / (1.0 + np.var(intervals))
    return float(consistency)


# ========================== VISUALIZATION METHODS ==========================

def create_ultimate_visualization(self, text: str, save_path: str = None) -> None:
    """Create comprehensive visualization of all analysis aspects."""
    stats = self.calculate_comprehensive_stats(text)
    positions = self.get_word_positions(text)

    if len(positions) == 0:
        print("No positions to visualize")
        return

    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 18))

    # 1. 3D Path with Quantum Field
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    self._plot_3d_path_with_field(ax1, positions, text)

    # 2. Comprehensive Metrics Radar
    ax2 = fig.add_subplot(3, 4, 2, projection='polar')
    self._plot_comprehensive_radar(ax2, stats)

    # 3. Finger Usage Heatmap
    ax3 = fig.add_subplot(3, 4, 3)
    self._plot_finger_heatmap(ax3, stats.finger_utilization)

    # 4. Harmonic Spectrum
    ax4 = fig.add_subplot(3, 4, 4)
    self._plot_harmonic_spectrum(ax4, text)

    # 5. PCA Analysis
    ax5 = fig.add_subplot(3, 4, 5)
    self._plot_pca_analysis(ax5, positions)

    # 6. Velocity Profile
    ax6 = fig.add_subplot(3, 4, 6)
    self._plot_velocity_profile(ax6, stats.velocity_profile)

    # 7. Quantum Coherence Evolution
    ax7 = fig.add_subplot(3, 4, 7)
    self._plot_quantum_evolution(ax7, positions)

    # 8. Information Metrics
    ax8 = fig.add_subplot(3, 4, 8)
    self._plot_information_metrics(ax8, stats)

    # 9. Efficiency Comparison
    ax9 = fig.add_subplot(3, 4, 9)
    self._plot_efficiency_metrics(ax9, stats)

    # 10. Biomechanical Load
    ax10 = fig.add_subplot(3, 4, 10)
    self._plot_biomechanical_analysis(ax10, text)

    # 11. Pattern Analysis
    ax11 = fig.add_subplot(3, 4, 11)
    self._plot_pattern_analysis(ax11, stats)

    # 12. Summary Statistics
    ax12 = fig.add_subplot(3, 4, 12)
    self._plot_summary_stats(ax12, stats)

    plt.suptitle(f'Ultimate Quantum Analysis: "{text}"', fontsize=20, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def _plot_3d_path_with_field(self, ax, positions, text):
    """Plot 3D path with quantum field visualization."""
    # Sample quantum field
    if hasattr(self, 'quantum_field'):
        field_sample = self.quantum_field[::3, ::2, ::1, 0]  # Use first time slice
        x_field = np.linspace(0, 15, field_sample.shape[0])
        y_field = np.linspace(0, 5, field_sample.shape[1])
        z_field = np.linspace(0, 2, field_sample.shape[2])

        X, Y, Z = np.meshgrid(x_field, y_field, z_field, indexing='ij')
        field_flat = field_sample.flatten()

        # Show high intensity points
        high_intensity = field_flat > np.percentile(field_flat, 95)
        if np.any(high_intensity):
            ax.scatter(X.flatten()[high_intensity],
                       Y.flatten()[high_intensity],
                       Z.flatten()[high_intensity],
                       c=field_flat[high_intensity],
                       cmap='plasma', alpha=0.1, s=2)

    # Plot typing path
    if len(positions) > 0:
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        scatter = ax.scatter(x, y, z, c=range(len(positions)),
                             cmap='viridis', s=100, alpha=0.8)

        if len(positions) > 1:
            ax.plot3D(x, y, z, 'b-', linewidth=3, alpha=0.7)

        # Add character labels
        for i, (pos, char) in enumerate(zip(positions, text)):
            ax.text(pos[0], pos[1], pos[2], char.upper(),
                    fontsize=10, weight='bold')

    ax.set_title('3D Quantum Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def _plot_comprehensive_radar(self, ax, stats):
    """Plot comprehensive metrics radar chart."""
    metrics = {
        'Distance': min(stats.total_distance / 20, 1.0),
        'Efficiency': (stats.bigram_efficiency + stats.trigram_efficiency) / 2,
        'Quantum Coherence': stats.quantum_coherence,
        'Hand Alternation': stats.hand_alternation_rate / 100,
        'Smoothness': stats.path_smoothness,
        'Information': stats.entropy / 5,
        'Rhythm': stats.rhythm_consistency,
        'Biomechanical': max(0, 1 - stats.biomechanical_load / 10)
    }

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    values = list(metrics.values())

    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics.keys(), fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Metrics', fontsize=12, pad=20)


def _plot_finger_heatmap(self, ax, finger_utilization):
    """Plot finger usage heatmap."""
    fingers = list(Finger)
    usage = [finger_utilization.get(f.value, 0) for f in fingers]

    # Create heatmap data
    left_fingers = [f for f in fingers if f.value.startswith('left')]
    right_fingers = [f for f in fingers if f.value.startswith('right')]

    left_usage = [finger_utilization.get(f.value, 0) for f in left_fingers]
    right_usage = [finger_utilization.get(f.value, 0) for f in right_fingers]

    # Combine for visualization
    all_usage = left_usage + right_usage
    finger_names = [f.value.split('_')[1] for f in left_fingers + right_fingers]

    bars = ax.bar(finger_names, all_usage,
                  color=['red' if i < len(left_usage) else 'blue'
                         for i in range(len(all_usage))])

    ax.set_title('Finger Utilization')
    ax.set_ylabel('Usage Count')
    ax.tick_params(axis='x', rotation=45)


def _plot_harmonic_spectrum(self, ax, text):
    """Plot harmonic frequency spectrum."""
    frequencies = []
    labels = []

    for char in text.lower():
        if char in self.layout.key_info and self.layout.key_info[char].harmonics:
            harmonics = self.layout.key_info[char].harmonics[:3]
            frequencies.extend(harmonics)
            labels.extend([f'{char.upper()}_{i + 1}' for i in range(len(harmonics))])

    if frequencies:
        bars = ax.bar(range(len(frequencies)), frequencies,
                      color=plt.cm.viridis(np.linspace(0, 1, len(frequencies))))
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Harmonic Spectrum')

        # Only show every nth label to avoid crowding
        step = max(1, len(labels) // 10)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=45, ha='right')


def _plot_pca_analysis(self, ax, positions):
    """Plot PCA analysis of positions."""
    if len(positions) >= 3:
        try:
            pca = PCA(n_components=2)
            projected = pca.fit_transform(positions)

            scatter = ax.scatter(projected[:, 0], projected[:, 1],
                                 c=range(len(positions)), cmap='viridis', s=60)

            if len(projected) > 1:
                ax.plot(projected[:, 0], projected[:, 1], 'b-', alpha=0.5)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_title('PCA Analysis')
        except:
            ax.text(0.5, 0.5, 'PCA Failed', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA Analysis (Failed)')
    else:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('PCA Analysis')


def _plot_velocity_profile(self, ax, velocity_profile):
    """Plot velocity profile over time."""
    if velocity_profile:
        ax.plot(velocity_profile, 'g-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(range(len(velocity_profile)), velocity_profile, alpha=0.3, color='green')
        ax.set_xlabel('Keystroke')
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Profile')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Velocity Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Velocity Profile')


def _plot_quantum_evolution(self, ax, positions):
    """Plot quantum state evolution."""
    if len(positions) > 0:
        quantum_states = []
        for i, pos in enumerate(positions):
            # Calculate evolving quantum state
            state = np.sin(pos[0] * 0.3 + i * 0.1) * np.cos(pos[1] * 0.5)
            quantum_states.append(state)

        ax.plot(quantum_states, 'purple', linewidth=2, marker='o')
        ax.fill_between(range(len(quantum_states)), quantum_states, alpha=0.3, color='purple')
        ax.set_xlabel('Position Index')
        ax.set_ylabel('Quantum State')
        ax.set_title('Quantum Evolution')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Quantum Evolution')


def _plot_information_metrics(self, ax, stats):
    """Plot information theory metrics."""
    metrics = {
        'Entropy': stats.entropy,
        'Mutual Info': stats.mutual_information,
        'Kolmogorov': stats.kolmogorov_complexity,
        'N-gram Entropy': stats.ngram_entropy
    }

    bars = ax.bar(metrics.keys(), metrics.values(), color=['red', 'blue', 'green', 'orange'])
    ax.set_title('Information Metrics')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)


def _plot_efficiency_metrics(self, ax, stats):
    """Plot efficiency metrics comparison."""
    metrics = {
        'Bigram': stats.bigram_efficiency,
        'Trigram': stats.trigram_efficiency,
        'Hand Alt': stats.hand_alternation_rate / 100,
        'Smoothness': stats.path_smoothness,
        'Rhythm': stats.rhythm_consistency
    }

    bars = ax.bar(metrics.keys(), metrics.values(),
                  color=plt.cm.Set3(np.linspace(0, 1, len(metrics))))
    ax.set_title('Efficiency Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)


def _plot_biomechanical_analysis(self, ax, text):
    """Plot biomechanical stress analysis."""
    stress_values = []
    char_labels = []

    for char in text.lower():
        if char in self.layout.key_info:
            stress = self.layout.key_info[char].biomechanical_stress
            stress_values.append(stress)
            char_labels.append(char.upper())

    if stress_values:
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(stress_values)))
        bars = ax.bar(range(len(stress_values)), stress_values, color=colors)

        ax.set_xlabel('Character')
        ax.set_ylabel('Biomechanical Stress')
        ax.set_title('Biomechanical Load')
        ax.set_xticks(range(len(char_labels)))
        ax.set_xticklabels(char_labels)

        # Add horizontal line for average
        if len(stress_values) > 1:
            avg_stress = np.mean(stress_values)
            ax.axhline(y=avg_stress, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_stress:.2f}')
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Biomechanical Load')


def _plot_pattern_analysis(self, ax, stats):
    """Plot pattern analysis metrics."""
    patterns = {
        'Bigram Eff': stats.bigram_efficiency,
        'Trigram Eff': stats.trigram_efficiency,
        'N-gram Entropy': stats.ngram_entropy / 5,  # Normalize
        'Rhythm Consistency': stats.rhythm_consistency
    }

    # Create pie chart
    values = list(patterns.values())
    labels = list(patterns.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(values)))

    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.2f', startangle=90)
    ax.set_title('Pattern Analysis')

    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')


def _plot_summary_stats(self, ax, stats):
    """Plot summary statistics table."""
    ax.axis('off')

    # Key statistics to display
    summary_data = [
        ['Metric', 'Value'],
        ['Total Distance', f'{stats.total_distance:.2f}'],
        ['Hand Alternation', f'{stats.hand_alternation_rate:.1f}%'],
        ['Quantum Coherence', f'{stats.quantum_coherence:.3f}'],
        ['Harmonic Resonance', f'{stats.harmonic_resonance:.3f}'],
        ['Information Entropy', f'{stats.entropy:.2f}'],
        ['Bigram Efficiency', f'{stats.bigram_efficiency:.3f}'],
        ['Path Smoothness', f'{stats.path_smoothness:.3f}'],
        ['Biomech Load', f'{stats.biomechanical_load:.2f}'],
        ['Fractal Dimension', f'{stats.dimensional_complexity:.2f}']
    ]

    # Create table
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(summary_data[0])):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Summary Statistics', fontweight='bold', pad=20)


# ========================== COMPARATIVE ANALYSIS ==========================

def compare_layouts(self, texts: List[str], layouts: List[str] = None) -> Dict[str, Dict[str, StatisticalSummary]]:
    """Compare multiple keyboard layouts with comprehensive statistics."""
    if layouts is None:
        layouts = ['qwerty', 'dvorak', 'colemak']

    results = {}

    for layout_name in layouts:
        print(f"Analyzing {layout_name.upper()} layout...")

        keyboard = UltimateQuantumKeyboard(layout_name)
        layout_metrics = defaultdict(list)

        for text in texts:
            stats = keyboard.calculate_comprehensive_stats(text)

            # Collect all numeric metrics
            for key, value in asdict(stats).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    layout_metrics[key].append(value)

        # Calculate statistical summaries
        results[layout_name] = {}
        for metric_name, values in layout_metrics.items():
            if values and len(values) > 0:
                results[layout_name][metric_name] = self._calculate_statistical_summary(values)

    return results


def _calculate_statistical_summary(self, values: List[float]) -> StatisticalSummary:
    """Calculate comprehensive statistical summary."""
    if not values:
        return StatisticalSummary(0, 0, 0, 0, 0, 0, 0, 0, (0, 0), (0, 0, 0), 0, 0.5)

    # Basic statistics
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

    # Advanced statistics
    try:
        # Skewness and kurtosis
        n = len(values)
        if n >= 3 and std_dev > 0:
            mean_centered = [(x - mean_val) for x in values]
            skewness = (n / ((n - 1) * (n - 2))) * sum((x / std_dev) ** 3 for x in mean_centered)

            if n >= 4:
                kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(
                    (x / std_dev) ** 4 for x in mean_centered) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
            else:
                kurtosis = 0.0
        else:
            skewness = 0.0
            kurtosis = 0.0
    except:
        skewness = 0.0
        kurtosis = 0.0

    # Confidence interval
    if len(values) > 1:
        margin_error = 1.96 * (std_dev / math.sqrt(len(values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
    else:
        ci_lower = ci_upper = mean_val

    # Quartiles
    try:
        q1 = statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min_val
        q2 = median_val
        q3 = statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max_val
    except:
        q1, q2, q3 = min_val, median_val, max_val

    # Outlier detection (IQR method)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = sum(1 for v in values if v < lower_bound or v > upper_bound)

    # Normality test (simplified Shapiro-Wilk approximation)
    try:
        if len(values) >= 3:
            # Simple normality check using skewness and kurtosis
            normality_score = abs(skewness) + abs(kurtosis) / 4
            normality_p = max(0.01, 1.0 / (1.0 + normality_score))
        else:
            normality_p = 0.5
    except:
        normality_p = 0.5

    return StatisticalSummary(
        mean=mean_val,
        median=median_val,
        std_dev=std_dev,
        variance=variance,
        min_value=min_val,
        max_value=max_val,
        skewness=skewness,
        kurtosis=kurtosis,
        confidence_interval_95=(ci_lower, ci_upper),
        quartiles=(q1, q2, q3),
        outlier_count=outlier_count,
        normality_test_p=normality_p
    )


def generate_comprehensive_report(self, texts: List[str], layouts: List[str] = None,
                                  output_file: str = None) -> str:
    """Generate a comprehensive analysis report."""
    if layouts is None:
        layouts = ['qwerty', 'dvorak', 'colemak']

    # Perform comparative analysis
    comparison_results = self.compare_layouts(texts, layouts)

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("ULTIMATE QUANTUM KEYBOARD ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Texts Analyzed: {len(texts)}")
    report.append(f"Layouts Compared: {', '.join(layouts).upper()}")
    report.append("")

    # Key findings
    report.append("KEY FINDINGS:")
    report.append("-" * 40)

    # Find best layout for each metric
    key_metrics = ['total_distance', 'hand_alternation_rate', 'quantum_coherence',
                   'bigram_efficiency', 'biomechanical_load']

    for metric in key_metrics:
        if all(metric in comparison_results[layout] for layout in layouts):
            best_layout = min(layouts, key=lambda l: comparison_results[l][metric].mean
            if 'load' in metric or 'distance' in metric
            else -comparison_results[l][metric].mean)

            best_value = comparison_results[best_layout][metric].mean
            report.append(f"Best {metric.replace('_', ' ').title()}: {best_layout.upper()} ({best_value:.3f})")

    report.append("")

    # Detailed statistics for each layout
    for layout in layouts:
        report.append(f"{layout.upper()} LAYOUT ANALYSIS:")
        report.append("-" * 40)

        if layout in comparison_results:
            layout_results = comparison_results[layout]

            # Top metrics
            important_metrics = [
                'total_distance', 'hand_alternation_rate', 'quantum_coherence',
                'bigram_efficiency', 'harmonic_resonance', 'entropy',
                'biomechanical_load', 'path_smoothness'
            ]

            for metric in important_metrics:
                if metric in layout_results:
                    stats = layout_results[metric]
                    report.append(f"{metric.replace('_', ' ').title():25}: "
                                  f"{stats.mean:8.3f} ± {stats.std_dev:6.3f} "
                                  f"[{stats.min_value:.3f}, {stats.max_value:.3f}]")

            report.append("")

    # Overall recommendations
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)

    # Calculate composite scores
    layout_scores = {}
    weights = {
        'total_distance': -1.0,
        'hand_alternation_rate': 1.0,
        'quantum_coherence': 2.0,
        'bigram_efficiency': 1.5,
        'harmonic_resonance': 1.0,
        'biomechanical_load': -1.5,
        'path_smoothness': 1.0
    }

    for layout in layouts:
        if layout in comparison_results:
            score = 0.0
            for metric, weight in weights.items():
                if metric in comparison_results[layout]:
                    metric_value = comparison_results[layout][metric].mean
                    score += weight * metric_value
            layout_scores[layout] = score

    # Sort by score
    ranked_layouts = sorted(layout_scores.items(), key=lambda x: x[1], reverse=True)

    report.append(f"1. OPTIMAL LAYOUT: {ranked_layouts[0][0].upper()}")
    report.append(f"   Composite Score: {ranked_layouts[0][1]:.3f}")
    report.append(f"   Best for: Quantum coherence, efficiency, ergonomics")
    report.append("")

    for i, (layout, score) in enumerate(ranked_layouts[1:], 2):
        report.append(f"{i}. {layout.upper()}: Score {score:.3f}")

    report.append("")
    report.append("TECHNICAL NOTES:")
    report.append("-" * 40)
    report.append("• Quantum metrics provide novel insights into typing flow")
    report.append("• Biomechanical analysis considers finger strength and positioning")
    report.append("• Information theory metrics reveal pattern complexity")
    report.append("• Machine learning features enable advanced pattern recognition")
    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")

    return report_text


# ========================== CONVENIENCE FUNCTIONS ==========================

def analyze_single_word(word: str, layout: str = 'qwerty') -> ComprehensiveTypingStats:
    """Quick analysis of a single word."""
    keyboard = UltimateQuantumKeyboard(layout)
    return keyboard.calculate_comprehensive_stats(word)


def compare_word_across_layouts(word: str, layouts: List[str] = None) -> Dict[str, ComprehensiveTypingStats]:
    """Compare a single word across multiple layouts."""
    if layouts is None:
        layouts = ['qwerty', 'dvorak', 'colemak']

    results = {}
    for layout in layouts:
        keyboard = UltimateQuantumKeyboard(layout)
        results[layout] = keyboard.calculate_comprehensive_stats(word)

    return results


def batch_analyze_texts(texts: List[str], layout: str = 'qwerty') -> List[ComprehensiveTypingStats]:
    """Analyze multiple texts with the same layout."""
    keyboard = UltimateQuantumKeyboard(layout)
    return [keyboard.calculate_comprehensive_stats(text) for text in texts]


def export_analysis_to_json(analysis_results: Dict, filename: str) -> None:
    """Export analysis results to JSON file."""

    def serialize_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(serialize_for_json(item) for item in obj)
        else:
            return obj

    serializable_data = serialize_for_json(analysis_results)

    try:
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"Analysis exported to {filename}")
    except Exception as e:
        print(f"Error exporting analysis: {e}")


# ========================== MAIN DEMONSTRATION ==========================

def main():
    """Demonstrate the Ultimate Quantum Keyboard Analyzer."""
    print("🚀 ULTIMATE QUANTUM KEYBOARD ANALYZER 🚀")
    print("=" * 60)

    # Sample texts for comprehensive analysis
    demo_texts = [
        "quantum computing",
        "keyboard efficiency analysis",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning",
        "biomechanical ergonomic optimization",
        "information theory entropy calculation",
        "harmonic resonance frequency patterns",
        "dimensional complexity fractal analysis",
        "neural network pattern recognition",
        "statistical significance testing methodology"
    ]

    # Single word demonstration
    print("\n1. SINGLE WORD QUANTUM ANALYSIS")
    print("-" * 40)

    demo_word = "quantum"
    print(f"Analyzing: '{demo_word}'")

    qwerty_kb = UltimateQuantumKeyboard('qwerty')
    stats = qwerty_kb.calculate_comprehensive_stats(demo_word)

    print(f"Total Distance: {stats.total_distance:.3f}")
    print(f"Quantum Coherence: {stats.quantum_coherence:.3f}")
    print(f"Harmonic Resonance: {stats.harmonic_resonance:.3f}")
    print(f"Bigram Efficiency: {stats.bigram_efficiency:.3f}")
    print(f"Information Entropy: {stats.entropy:.3f}")
    print(f"Biomechanical Load: {stats.biomechanical_load:.3f}")

    # Layout comparison
    print("\n2. MULTI-LAYOUT COMPARISON")
    print("-" * 40)

    layouts = ['qwerty', 'dvorak', 'colemak']
    comparison_word = "efficiency"

    print(f"Comparing '{comparison_word}' across layouts:")

    for layout in layouts:
        kb = UltimateQuantumKeyboard(layout)
        stats = kb.calculate_comprehensive_stats(comparison_word)

        print(f"{layout.upper():8}: Distance={stats.total_distance:.2f}, "
              f"Quantum={stats.quantum_coherence:.3f}, "
              f"Efficiency={stats.bigram_efficiency:.3f}")

    # Comprehensive analysis
    print("\n3. COMPREHENSIVE ANALYSIS")
    print("-" * 40)

    print("Performing comprehensive analysis across all layouts...")
    results = qwerty_kb.compare_layouts(demo_texts[:5], layouts)

    # Find best layout for key metrics
    key_metrics = ['total_distance', 'quantum_coherence', 'bigram_efficiency']

    for metric in key_metrics:
        best_layout = min(layouts,
                          key=lambda l: results[l][metric].mean if metric == 'total_distance'
                          else -results[l][metric].mean)
        best_value = results[best_layout][metric].mean

        print(f"Best {metric.replace('_', ' ')}: {best_layout.upper()} ({best_value:.3f})")

    # Generate report
    print("\n4. GENERATING COMPREHENSIVE REPORT")
    print("-" * 40)

    report = qwerty_kb.generate_comprehensive_report(demo_texts[:3], layouts)

    # Show abbreviated report
    report_lines = report.split('\n')
    for line in report_lines[:30]:  # Show first 30 lines
        print(line)

    if len(report_lines) > 30:
        print("... (report continues)")

    # Visualization demo
    print("\n5. VISUALIZATION CAPABILITIES")
    print("-" * 40)

    try:
        print("Generating ultimate visualization...")
        qwerty_kb.create_ultimate_visualization(demo_word)
        print("✓ Visualization complete!")
    except Exception as e:
        print(f"Visualization requires additional packages: {e}")
        print("Install: pip install matplotlib seaborn scikit-learn")

    # Performance benchmark
    print("\n6. PERFORMANCE BENCHMARK")
    print("-" * 40)

    start_time = time.time()
    for text in demo_texts[:5]:
        stats = qwerty_kb.calculate_comprehensive_stats(text)
    analysis_time = time.time() - start_time

    print(f"Analyzed {len(demo_texts[:5])} texts in {analysis_time:.3f} seconds")
    print(f"Average time per text: {analysis_time / 5:.3f} seconds")

    # Export demonstration
    print("\n7. EXPORT CAPABILITIES")
    print("-" * 40)

    try:
        sample_results = {
            'qwerty': analyze_single_word(demo_word, 'qwerty'),
            'dvorak': analyze_single_word(demo_word, 'dvorak')
        }

        export_analysis_to_json(sample_results, 'quantum_analysis_demo.json')
        print("✓ Analysis exported to JSON")
    except Exception as e:
        print(f"Export demo failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 ULTIMATE QUANTUM KEYBOARD ANALYSIS COMPLETE! 🎯")
    print("Features demonstrated:")
    print("• 30+ comprehensive metrics")
    print("• Quantum-inspired analysis")
    print("• Machine learning integration")
    print("• Statistical rigor with confidence intervals")
    print("• Advanced visualization capabilities")
    print("• Multi-layout comparative analysis")
    print("• Biomechanical and ergonomic modeling")
    print("• Information theory metrics")
    print("• Pattern recognition and efficiency analysis")
    print("• Comprehensive reporting and export")


if __name__ == "__main__":
    main()  # ultimate_quantum_keyboard.py

import numpy as np
import zlib
import gzip
import json
import io
import math
import statistics
import time
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ========================== ENUMS AND DATA CLASSES ==========================

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


class InputType(Enum):
    KEYBOARD = "keyboard"
    TOUCH = "touch"
    VR = "vr"
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"


@dataclass
class QuantumKeyInfo:
    """Enhanced key information with quantum and ergonomic properties."""
    position: np.ndarray
    finger: Finger
    hand: Hand
    effort: float = 1.0
    frequency: float = 0.0  # English letter frequency
    quantum_state: float = 0.0
    harmonics: List[float] = None
    biomechanical_stress: float = 1.0
    neural_activation: float = 1.0

    def __post_init__(self):
        if self.harmonics is None:
            self.harmonics = []


@dataclass
class ComprehensiveTypingStats:
    """Ultimate typing statistics with all metrics."""
    # Basic geometric metrics
    total_distance: float
    avg_distance_per_char: float
    path_variance: float

    # Ergonomic metrics
    hand_alternation_rate: float
    finger_utilization: Dict[str, int]  # Use string keys for JSON serialization
    same_finger_percentage: float
    total_effort: float
    biomechanical_load: float

    # Advanced geometric metrics
    curvature: float
    torsion: float
    planarity: float
    compactness: float
    path_smoothness: float

    # Quantum-inspired metrics
    quantum_coherence: float
    harmonic_resonance: float
    dimensional_complexity: float
    quantum_entanglement: float
    phase_synchronization: float

    # Information theory metrics
    entropy: float
    mutual_information: float
    kolmogorov_complexity: float

    # Pattern analysis
    bigram_efficiency: float
    trigram_efficiency: float
    ngram_entropy: float
    rhythm_consistency: float

    # Machine learning features
    pca_variance_explained: float
    clustering_quality: float
    anomaly_score: float

    # Temporal metrics
    typing_rhythm: float
    acceleration_variance: float
    velocity_profile: List[float] = None

    def __post_init__(self):
        if self.velocity_profile is None:
            self.velocity_profile = []


@dataclass
class StatisticalSummary:
    """Enhanced statistical analysis with advanced metrics."""
    mean: float
    median: float
    std_dev: float
    variance: float
    min_value: float
    max_value: float
    skewness: float
    kurtosis: float
    confidence_interval_95: Tuple[float, float]
    quartiles: Tuple[float, float, float]  # Q1, Q2, Q3
    outlier_count: int
    normality_test_p: float


# ========================== KEYBOARD LAYOUTS ==========================

class KeyboardLayout(ABC):
    """Abstract base class for all keyboard layouts."""

    def __init__(self, layout_name: str):
        self.layout_name = layout_name
        self.key_info: Dict[str, QuantumKeyInfo] = {}
        self.setup_layout()

    @abstractmethod
    def setup_layout(self) -> None:
        """Setup the specific keyboard layout."""
        pass

    def add_key(self, char: str, x: float, y: float, z: float = 0.0,
                finger: Finger = Finger.RIGHT_INDEX, effort: float = 1.0,
                frequency: float = 0.0) -> None:
        """Add a key to the layout with comprehensive information."""
        hand = Hand.LEFT if finger.value.startswith('left') else Hand.RIGHT

        # Calculate quantum properties
        quantum_state = self._calculate_quantum_state(x, y, z, frequency)
        harmonics = self._generate_harmonics(x, y, z)
        biomech_stress = self._calculate_biomechanical_stress(x, y, z, finger)
        neural_activation = self._calculate_neural_activation(frequency, effort)

        self.key_info[char.lower()] = QuantumKeyInfo(
            position=np.array([x, y, z]),
            finger=finger,
            hand=hand,
            effort=effort,
            frequency=frequency,
            quantum_state=quantum_state,
            harmonics=harmonics,
            biomechanical_stress=biomech_stress,
            neural_activation=neural_activation
        )

    def _calculate_quantum_state(self, x: float, y: float, z: float, freq: float) -> float:
        """Calculate quantum state based on position and frequency."""
        spatial = np.sin(x * 0.3) * np.cos(y * 0.5) * np.exp(-z * 0.2)
        frequency_component = freq / 100.0
        return float((spatial + frequency_component) / 2.0)

    def _generate_harmonics(self, x: float, y: float, z: float) -> List[float]:
        """Generate harmonic frequencies for position."""
        base_freq = 440.0 * (2 ** ((x + y * 12 + z * 144) / 12))
        return [base_freq * i for i in range(1, 6)]

    def _calculate_biomechanical_stress(self, x: float, y: float, z: float, finger: Finger) -> float:
        """Calculate biomechanical stress based on position and finger."""
        # Distance from home row
        home_row_y = 2.0
        distance_penalty = abs(y - home_row_y) * 0.2

        # Finger-specific penalties
        finger_penalties = {
            Finger.LEFT_PINKY: 1.3, Finger.RIGHT_PINKY: 1.3,
            Finger.LEFT_RING: 1.1, Finger.RIGHT_RING: 1.1,
            Finger.LEFT_MIDDLE: 0.9, Finger.RIGHT_MIDDLE: 0.9,
            Finger.LEFT_INDEX: 0.8, Finger.RIGHT_INDEX: 0.8,
            Finger.LEFT_THUMB: 0.6, Finger.RIGHT_THUMB: 0.6
        }

        base_stress = finger_penalties.get(finger, 1.0)
        return base_stress + distance_penalty + z * 0.3

    def _calculate_neural_activation(self, frequency: float, effort: float) -> float:
        """Calculate neural activation based on frequency and effort."""
        freq_activation = np.log1p(frequency) / 5.0  # Logarithmic scaling
        effort_activation = effort
        return (freq_activation + effort_activation) / 2.0


class QWERTYLayout(KeyboardLayout):
    """Complete QWERTY layout with all keys."""

    def setup_layout(self) -> None:
        # Letter frequencies in English
        frequencies = {
            'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75,
            's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
            'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }

        # Number row
        numbers = "1234567890"
        number_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                          Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                          Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                          Finger.RIGHT_PINKY]

        for i, (num, finger) in enumerate(zip(numbers, number_fingers)):
            self.add_key(num, i + 1, 0, 0, finger, 1.2 + i * 0.05, 0.1)

        # Top row (QWERTY)
        top_row = "qwertyuiop"
        top_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                       Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                       Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                       Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(top_row, top_fingers)):
            freq = frequencies.get(char, 0.0)
            effort = 1.0 + i * 0.02
            self.add_key(char, i, 1, 0, finger, effort, freq)

        # Home row (ASDF...)
        home_row = "asdfghjkl;"
        home_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                        Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                        Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                        Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(home_row, home_fingers)):
            freq = frequencies.get(char, 0.0)
            effort = 0.6 + i * 0.02  # Home row is most comfortable
            self.add_key(char, i, 2, 0, finger, effort, freq)

        # Bottom row (ZXCV...)
        bottom_row = "zxcvbnm,./"
        bottom_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                          Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                          Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                          Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(bottom_row, bottom_fingers)):
            freq = frequencies.get(char, 0.0)
            effort = 1.1 + i * 0.02
            self.add_key(char, i, 3, 0, finger, effort, freq)

        # Space bar
        self.add_key(' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5, 18.0)


class DvorakLayout(KeyboardLayout):
    """Dvorak keyboard layout optimized for efficiency."""

    def setup_layout(self) -> None:
        frequencies = {
            'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75,
            's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
            'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }

        # Dvorak layout
        # Top row
        top_row = "\",.pyfgcrl"
        top_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                       Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                       Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                       Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(top_row, top_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 1, 0, finger, 0.9 + i * 0.02, freq)

        # Home row (vowels optimized)
        home_row = "aoeuidhtns"
        home_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                        Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                        Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                        Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(home_row, home_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 2, 0, finger, 0.6 + i * 0.02, freq)

        # Bottom row
        bottom_row = ";qjkxbmwvz"
        bottom_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                          Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                          Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                          Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(bottom_row, bottom_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 3, 0, finger, 1.0 + i * 0.02, freq)

        # Space bar
        self.add_key(' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5, 18.0)


class ColemakLayout(KeyboardLayout):
    """Colemak keyboard layout - modern alternative."""

    def setup_layout(self) -> None:
        frequencies = {
            'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75,
            's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
            'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }

        # Colemak layout
        # Top row
        top_row = "qwfpgjluy;"
        top_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                       Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                       Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                       Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(top_row, top_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 1, 0, finger, 0.9 + i * 0.02, freq)

        # Home row (optimized)
        home_row = "arstdhneio"
        home_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                        Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                        Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                        Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(home_row, home_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 2, 0, finger, 0.6 + i * 0.02, freq)

        # Bottom row
        bottom_row = "zxcvbkm,./"
        bottom_fingers = [Finger.LEFT_PINKY, Finger.LEFT_RING, Finger.LEFT_MIDDLE,
                          Finger.LEFT_INDEX, Finger.LEFT_INDEX, Finger.RIGHT_INDEX,
                          Finger.RIGHT_INDEX, Finger.RIGHT_MIDDLE, Finger.RIGHT_RING,
                          Finger.RIGHT_PINKY]

        for i, (char, finger) in enumerate(zip(bottom_row, bottom_fingers)):
            freq = frequencies.get(char, 0.0)
            self.add_key(char, i, 3, 0, finger, 1.0 + i * 0.02, freq)

        # Space bar
        self.add_key(' ', 4.5, 4, 0, Finger.RIGHT_THUMB, 0.5, 18.0)


# ========================== MAIN QUANTUM KEYBOARD CLASS ==========================

class UltimateQuantumKeyboard:
    """
    Ultimate quantum keyboard analyzer combining all advanced features.

    This class represents the pinnacle of keyboard analysis, incorporating:
    - Traditional geometric analysis
    - Quantum-inspired metrics
    - Machine learning techniques
    - Advanced statistical analysis
    - Biomechanical modeling
    - Neural network features
    """

    def __init__(self, layout: Union[str, KeyboardLayout] = 'qwerty'):
        """Initialize the ultimate quantum keyboard."""
        if isinstance(layout, str):
            layout_classes = {
                'qwerty': QWERTYLayout,
                'dvorak': DvorakLayout,
                'colemak': ColemakLayout
            }
            if layout.lower() in layout_classes:
                self.layout = layout_classes[layout.lower()](layout.lower())
            else:
                raise ValueError(f"Unknown layout: {layout}")
        else:
            self.layout = layout

        # Initialize caches and quantum field
        self.analysis_cache = {}
        self.quantum_field = self._initialize_quantum_field()
        self.neural_network = self._initialize_neural_features()

        # ML models
        self.pca_model = None
        self.scaler = StandardScaler()
        self.clustering_model = None

    def _initialize_quantum_field(self) -> np.ndarray:
        """Initialize 4D quantum field over keyboard space-time."""
        x = np.linspace(0, 15, 30)
        y = np.linspace(0, 5, 15)
        z = np.linspace(0, 2, 8)
        t = np.linspace(0, 1, 5)  # Time dimension

        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')

        # Multi-harmonic quantum field
        field = (np.sin(X * 0.5 + T * 2 * np.pi) * np.cos(Y * 0.8) * np.exp(-Z * 0.3) +
                 0.5 * np.sin(X * 1.2) * np.sin(Y * 1.5 + T * np.pi) * np.cos(Z * 2.0) +
                 0.3 * np.cos(X * 0.3 + Y * 0.7 + Z * 1.1) * np.sin(T * 3 * np.pi))

        return field

    def _initialize_neural_features(self) -> Dict[str, Any]:
        """Initialize neural network-inspired features."""
        return {
            'activation_patterns': defaultdict(list),
            'synaptic_weights': defaultdict(float),
            'neural_plasticity': 1.0,
            'learning_rate': 0.01
        }

    # ========================== CORE ANALYSIS METHODS ==========================

    def get_word_positions(self, word: str) -> np.ndarray:
        """Get 3D positions for each character in a word."""
        positions = []
        for char in word.lower():
            if char in self.layout.key_info:
                positions.append(self.layout.key_info[char].position)

        return np.array(positions) if positions else np.array([]).reshape(0, 3)

    def calculate_comprehensive_stats(self, text: str) -> ComprehensiveTypingStats:
        """Calculate all possible metrics for the text."""
        # Use cache if available
        cache_key = f"{self.layout.layout_name}_{hash(text)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        positions = self.get_word_positions(text)

        if len(positions) == 0:
            return self._empty_stats()

        # Calculate all metric categories
        basic_metrics = self._calculate_basic_metrics(positions, text)
        ergonomic_metrics = self._calculate_ergonomic_metrics(text)
        geometric_metrics = self._calculate_geometric_metrics(positions)
        quantum_metrics = self._calculate_quantum_metrics(positions, text)
        information_metrics = self._calculate_information_metrics(text, positions)
        pattern_metrics = self._calculate_pattern_metrics(text)
        ml_metrics = self._calculate_ml_metrics(positions, text)
        temporal_metrics = self._calculate_temporal_metrics(positions)

        # Combine all metrics
        stats = ComprehensiveTypingStats(
            **basic_metrics,
            **ergonomic_metrics,
            **geometric_metrics,
            **quantum_metrics,
            **information_metrics,
            **pattern_metrics,
            **ml_metrics,
            **temporal_metrics
        )

        # Cache the result
        self.analysis_cache[cache_key] = stats
        return stats

    def _empty_stats(self) -> ComprehensiveTypingStats:
        """Return empty statistics for invalid input."""
        return ComprehensiveTypingStats(
            total_distance=0.0, avg_distance_per_char=0.0, path_variance=0.0,
            hand_alternation_rate=0.0, finger_utilization={}, same_finger_percentage=0.0,
            total_effort=0.0, biomechanical_load=0.0, curvature=0.0, torsion=0.0,
            planarity=0.0, compactness=0.0, path_smoothness=0.0, quantum_coherence=0.0,
            harmonic_resonance=0.0, dimensional_complexity=0.0, quantum_entanglement=0.0,
            phase_synchronization=0.0, entropy=0.0, mutual_information=0.0,
            kolmogorov_complexity=0.0, bigram_efficiency=0.0, trigram_efficiency=0.0,
            ngram_entropy=0.0, rhythm_consistency=0.0, pca_variance_explained=0.0,
            clustering_quality=0.0, anomaly_score=0.0, typing_rhythm=0.0,
            acceleration_variance=0.0, velocity_profile=[]
        )

    def _calculate_basic_metrics(self, positions: np.ndarray, text: str) -> Dict[str, float]:
        """Calculate basic geometric metrics."""
        if len(positions) < 2:
            return {
                'total_distance': 0.0,
                'avg_distance_per_char': 0.0,
                'path_variance': 0.0
            }

        # Distance calculations
        diffs = positions[1:] - positions[:-1]
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = float(np.sum(distances))
        avg_distance = total_distance / max(1, len(text))
        path_variance = float(np.var(distances)) if len(distances) > 1 else 0.0

        return {
            'total_distance': total_distance,
            'avg_distance_per_char': avg_distance,
            'path_variance': path_variance
        }

    def _calculate_ergonomic_metrics(self, text: str) -> Dict[str, Union[float, Dict]]:
        """Calculate ergonomic and biomechanical metrics."""
        hand_sequence = []
        finger_sequence = []
        effort_sequence = []
        biomech_sequence = []

        for char in text.lower():
            if char in self.layout.key_info:
                key_info = self.layout.key_info[char]
                hand_sequence.append(key_info.hand)
                finger_sequence.append(key_info.finger)
                effort_sequence.append(key_info.effort)
                biomech_sequence.append(key_info.biomechanical_stress)

        # Hand alternation
        if len(hand_sequence) > 1:
            alternations = sum(1 for i in range(1, len(hand_sequence))
                               if hand_sequence[i] != hand_sequence[i - 1])
            hand_alternation_rate = (alternations / (len(hand_sequence) - 1)) * 100
        else:
            hand_alternation_rate = 0.0

        # Finger utilization
        finger_counts = Counter(finger_sequence)
        finger_utilization = {f.value: finger_counts.get(f, 0) for f in Finger}

        # Same finger percentage
        if len(finger_sequence) > 1:
            same_finger = sum(1 for i in range(1, len(finger_sequence))
                              if finger_sequence[i] == finger_sequence[i - 1])
            same_finger_percentage = (same_finger / (len(finger_sequence) - 1)) * 100
        else:
            same_finger_percentage = 0.0

        # Effort and biomechanical load
        total_effort = sum(effort_sequence)
        biomechanical_load = sum(biomech_sequence)

        return {
            'hand_alternation_rate': hand_alternation_rate,
            'finger_utilization': finger_utilization,
            'same_finger_percentage': same_finger_percentage,
            'total_effort': total_effort,
            'biomechanical_load': biomechanical_load
        }

    def _calculate_geometric_metrics(self, positions: np.ndarray) -> Dict[str, float]:
        """Calculate advanced geometric metrics."""
        if len(positions) < 3:
            return {
                'curvature': 0.0,
                'torsion': 0.0,
                'planarity': 0.0,
                'compactness': 0.0,
                'path_smoothness': 0.0
            }

        # Curvature calculation
        curvature = self._calculate_curvature(positions)

        # Torsion calculation
        torsion = self._calculate_torsion(positions)

        # Planarity using PCA
        try:
            pca = PCA(n_components=2)
            pca.fit(positions)
            planarity = float(sum(pca.explained_variance_ratio_))
        except:
            planarity = 0.0

        # Compactness
        compactness = self._calculate_compactness(positions)

        # Path smoothness (second derivative analysis)
        if len(positions) >= 3:
            first_deriv = positions[1:] - positions[:-1]
            second_deriv = first_deriv[1:] - first_deriv[:-1]
            smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(second_deriv, axis=1)))
        else:
            smoothness = 1.0

        return {
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_dtft_network():
    """
    Visualize the DTFT-based network architecture for multi-BD frequency estimation
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    color_input = '#E3F2FD'  # Light blue
    color_rnn = '#FFF3E0'    # Light orange
    color_dtft = '#F3E5F5'   # Light purple
    color_attention = '#E8F5E9'  # Light green
    color_output = '#FFEBEE'  # Light red
    
    # Title
    ax.text(5, 13.5, 'Multi-BD DTFT-Based Frequency Estimation Network', 
            ha='center', fontsize=16, fontweight='bold')
    
    # ==================== TIME LOOP (t = 0 to tau-1) ====================
    y_start = 12
    
    # Input at each time step
    box_input = FancyBboxPatch((0.2, y_start), 1.5, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box_input)
    ax.text(0.95, y_start + 0.4, 'Observation y(t)\n[N_ris×1]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Time step indicator
    ax.text(0.95, y_start - 0.3, 't = 0, 1, ..., τ-1', 
            ha='center', fontsize=8, style='italic')
    
    # BD modulation
    ax.text(0.95, y_start + 1.2, 'x_BD(t) ∈ {-1, +1}', 
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Carrier
    ax.text(0.95, y_start + 1.6, 's(t) = exp(j·2π·t/τ)', 
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))
    
    # Arrow to observation processing
    arrow1 = FancyArrowPatch((1.7, y_start + 0.4), (2.2, y_start + 0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Observation processing
    box_proc = FancyBboxPatch((2.2, y_start - 0.1), 1.5, 1.0, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box_proc)
    ax.text(2.95, y_start + 0.4, 'Real/Imag\nSeparation\n+ Features', 
            ha='center', va='center', fontsize=9)
    
    # Arrow to RNN
    arrow2 = FancyArrowPatch((3.7, y_start + 0.4), (4.2, y_start + 0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # LSTM/RNN block
    box_rnn = FancyBboxPatch((4.2, y_start - 0.2), 1.8, 1.2, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color_rnn, linewidth=2)
    ax.add_patch(box_rnn)
    ax.text(5.1, y_start + 0.7, 'LSTM Cell', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.1, y_start + 0.3, 'Hidden: h(t)\nCell: c(t)', 
            ha='center', va='center', fontsize=8)
    
    # Recurrent connection
    arrow_recur = FancyArrowPatch((5.5, y_start + 1.0), (6.5, y_start + 1.0),
                                 arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                                 color='red', linestyle='dashed')
    ax.add_patch(arrow_recur)
    ax.text(6.0, y_start + 1.3, 'h(t-1), c(t-1)', 
            ha='center', fontsize=7, color='red')
    
    # Beamformer output (optional, shown in lighter box)
    box_beam = FancyBboxPatch((6.2, y_start - 0.1), 1.3, 0.8, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='gray', facecolor='white', 
                              linewidth=1, linestyle='dashed', alpha=0.5)
    ax.add_patch(box_beam)
    ax.text(6.85, y_start + 0.3, 'v(t)\n(optional)', 
            ha='center', va='center', fontsize=8, style='italic', color='gray')
    
    # Time loop annotation
    ax.annotate('', xy=(0.2, y_start + 2.2), xytext=(7.5, y_start + 2.2),
                arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax.text(3.85, y_start + 2.5, 'Repeat for τ time steps', 
            ha='center', fontsize=10, fontweight='bold', color='blue')
    
    # ==================== AFTER τ STEPS: FREQUENCY ESTIMATION ====================
    y_dtft = 9
    
    # Collected observations
    box_obs = FancyBboxPatch((0.2, y_dtft), 1.5, 0.8, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box_obs)
    ax.text(0.95, y_dtft + 0.4, 'Observations\ny[0:τ-1]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # LSTM features
    box_lstm_feat = FancyBboxPatch((0.2, y_dtft - 1.2), 1.5, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='black', facecolor=color_rnn, linewidth=2)
    ax.add_patch(box_lstm_feat)
    ax.text(0.95, y_dtft - 0.8, 'LSTM State\n[h(τ), c(τ)]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow to DTFT
    arrow3 = FancyArrowPatch((1.7, y_dtft + 0.4), (2.2, y_dtft + 0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # DTFT Computation Block
    box_dtft = FancyBboxPatch((2.2, y_dtft - 0.3), 2.0, 1.4, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor=color_dtft, linewidth=2)
    ax.add_patch(box_dtft)
    ax.text(3.2, y_dtft + 0.8, 'DTFT Module', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.2, y_dtft + 0.4, 'X(ωₖ) = Σ y[t]·e^(-jωₖt)', 
            ha='center', va='center', fontsize=8)
    ax.text(3.2, y_dtft + 0.0, 'For ωₖ ∈ [0, 2π]', 
            ha='center', va='center', fontsize=8)
    ax.text(3.2, y_dtft - 0.2, 'K frequency bins', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Frequency candidates
    ax.text(3.2, y_dtft - 0.7, 'ω = [0, π/4, π/2, 3π/4, π, ...]', 
            ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='lavender'))
    
    # Arrow from LSTM to Attention
    arrow4 = FancyArrowPatch((1.7, y_dtft - 0.8), (2.2, y_dtft - 1.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow4)
    
    # Attention Network
    box_attn = FancyBboxPatch((2.2, y_dtft - 2.2), 2.0, 1.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor=color_attention, linewidth=2)
    ax.add_patch(box_attn)
    ax.text(3.2, y_dtft - 1.3, 'Attention Network', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.2, y_dtft - 1.6, 'MLP([h, c]) → α(ωₖ)', 
            ha='center', va='center', fontsize=8)
    ax.text(3.2, y_dtft - 1.9, 'Learned weights per freq', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Arrow to spectrum fusion
    arrow5 = FancyArrowPatch((4.2, y_dtft + 0.4), (4.7, y_dtft + 0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((4.2, y_dtft - 1.6), (4.7, y_dtft - 0.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow6)
    
    # Weighted Spectrum
    box_weighted = FancyBboxPatch((4.7, y_dtft - 0.3), 1.8, 1.4, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='black', facecolor=color_dtft, linewidth=2)
    ax.add_patch(box_weighted)
    ax.text(5.6, y_dtft + 0.7, 'Enhanced Spectrum', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.6, y_dtft + 0.3, 'P(ωₖ) = α(ωₖ)·|X(ωₖ)|²', 
            ha='center', va='center', fontsize=8)
    ax.text(5.6, y_dtft - 0.1, '[K frequency bins]', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Arrow to peak detection
    arrow7 = FancyArrowPatch((6.5, y_dtft + 0.4), (7.0, y_dtft + 0.4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    # Peak Detection / Top-K
    box_peak = FancyBboxPatch((7.0, y_dtft - 0.1), 1.8, 1.0, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(box_peak)
    ax.text(7.9, y_dtft + 0.6, 'Peak Detection', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.9, y_dtft + 0.2, 'Top-3 frequencies', 
            ha='center', va='center', fontsize=8)
    ax.text(7.9, y_dtft - 0.1, 'Soft argmax', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Final output
    box_output = FancyBboxPatch((7.0, y_dtft - 1.4), 1.8, 0.9, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(box_output)
    ax.text(7.9, y_dtft - 0.7, 'Detected Ω', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.9, y_dtft - 1.0, '[ω₁, ω₂, ω₃]', 
            ha='center', va='center', fontsize=9)
    ax.text(7.9, y_dtft - 1.25, '(1-3 active BDs)', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # ==================== SPECTRUM VISUALIZATION ====================
    y_spec = 5.5
    
    # Example spectrum plot
    ax_spec = fig.add_axes([0.15, 0.05, 0.7, 0.3])
    
    # Generate example spectrum with 3 peaks
    freq_bins = np.linspace(0, 2*np.pi, 64)
    spectrum = np.zeros_like(freq_bins)
    
    # Add 3 peaks at different frequencies
    peak1_idx = 8   # π/4
    peak2_idx = 16  # π/2
    peak3_idx = 32  # π
    
    for idx in [peak1_idx, peak2_idx, peak3_idx]:
        spectrum += 2.0 * np.exp(-0.5 * ((freq_bins - freq_bins[idx]) / 0.15)**2)
    
    # Add noise
    spectrum += 0.1 * np.random.randn(len(freq_bins))
    spectrum = np.maximum(spectrum, 0)
    
    ax_spec.plot(freq_bins / np.pi, spectrum, linewidth=2, color='purple')
    ax_spec.fill_between(freq_bins / np.pi, spectrum, alpha=0.3, color='purple')
    ax_spec.axvline(freq_bins[peak1_idx] / np.pi, color='red', linestyle='--', linewidth=2, label='BD #1: ω₁=π/4')
    ax_spec.axvline(freq_bins[peak2_idx] / np.pi, color='green', linestyle='--', linewidth=2, label='BD #2: ω₂=π/2')
    ax_spec.axvline(freq_bins[peak3_idx] / np.pi, color='blue', linestyle='--', linewidth=2, label='BD #3: ω₃=π')
    
    ax_spec.set_xlabel('Frequency (×π rad)', fontsize=11, fontweight='bold')
    ax_spec.set_ylabel('Spectral Power P(ω)', fontsize=11, fontweight='bold')
    ax_spec.set_title('Example: Multi-BD Frequency Spectrum (3 Active BDs)', fontsize=12, fontweight='bold')
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(loc='upper right', fontsize=9)
    ax_spec.set_xlim([0, 2])
    
    # ==================== ANNOTATIONS ====================
    
    # Key features box
    features_text = """
Key Features:
• DTFT basis: Physical frequency analysis
• LSTM features: Learned noise/interference suppression  
• Attention: Adaptive frequency weighting
• Top-K peaks: Multi-BD detection (1-3 simultaneous)
• Differentiable: End-to-end gradient flow
• Interpretable: Visualizable spectrum
    """
    
    ax.text(9.5, 11, features_text, 
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=1.5))
    
    # Training info
    training_text = """
Training Strategy:
Phase 1: Single BD (ω=π)
Phase 2: Multiple BDs (different ω)
Phase 3: Simultaneous transmission
    """
    
    ax.text(9.5, 8, training_text, 
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('dtft_network_architecture.png', dpi=300, bbox_inches='tight')
    print("Network architecture saved to 'dtft_network_architecture.png'")
    plt.show()

if __name__ == '__main__':
    visualize_dtft_network()
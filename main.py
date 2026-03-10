import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Generate random binary data
# ------------------------------------------------------------
def generate_bits(n_bits: int) -> np.ndarray:
    """Generate random binary bits (0 or 1)."""
    return np.random.randint(0, 2, n_bits)


# ------------------------------------------------------------
# QPSK Modulation
# Convert bit pairs into complex QPSK symbols using Gray coding
# ------------------------------------------------------------
def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    QPSK modulation using Gray coding.

    Mapping:
        00 -> (1 + 1j)/sqrt(2)
        01 -> (-1 + 1j)/sqrt(2)
        11 -> (-1 - 1j)/sqrt(2)
        10 -> (1 - 1j)/sqrt(2)
    """

    # QPSK requires pairs of bits
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK modulation.")

    # Reshape bit stream into pairs
    bit_pairs = bits.reshape(-1, 2)
    symbols = []

    # Map each pair of bits to a complex symbol
    for b1, b2 in bit_pairs:
        if b1 == 0 and b2 == 0:
            symbols.append((1 + 1j) / np.sqrt(2))
        elif b1 == 0 and b2 == 1:
            symbols.append((-1 + 1j) / np.sqrt(2))
        elif b1 == 1 and b2 == 1:
            symbols.append((-1 - 1j) / np.sqrt(2))
        elif b1 == 1 and b2 == 0:
            symbols.append((1 - 1j) / np.sqrt(2))

    return np.array(symbols)


# ------------------------------------------------------------
# AWGN Channel
# Add Gaussian noise to simulate a noisy communication channel
# ------------------------------------------------------------
def add_awgn_noise(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex AWGN noise to QPSK symbols.

    snr_db : Signal-to-noise ratio in decibels
    """

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise variance
    noise_variance = 1 / (2 * snr_linear)

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_variance) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )

    # Add noise to transmitted symbols
    return symbols + noise


# ------------------------------------------------------------
# QPSK Demodulation
# Convert received symbols back into bits
# ------------------------------------------------------------
def qpsk_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    """
    Hard decision demodulation for QPSK.

    Decision based on sign of real and imaginary parts.
    """

    demod_bits = []

    for symbol in received_symbols:

        # Extract real and imaginary parts
        real = np.real(symbol)
        imag = np.imag(symbol)

        # Determine symbol quadrant
        if real > 0 and imag > 0:
            demod_bits.extend([0, 0])
        elif real < 0 and imag > 0:
            demod_bits.extend([0, 1])
        elif real < 0 and imag < 0:
            demod_bits.extend([1, 1])
        else:
            demod_bits.extend([1, 0])

    return np.array(demod_bits)


# ------------------------------------------------------------
# Bit Error Rate Calculation
# ------------------------------------------------------------
def calculate_ber(original_bits: np.ndarray, received_bits: np.ndarray) -> float:
    """Calculate Bit Error Rate (BER)."""

    # Count bit errors
    errors = np.sum(original_bits != received_bits)

    # BER = number of errors / total bits
    return errors / len(original_bits)


# ------------------------------------------------------------
# Main Simulation
# ------------------------------------------------------------
def main():

    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of transmitted bits
    n_bits = 100000

    # SNR range in dB
    snr_db_values = np.arange(0, 13, 1)

    # Store BER results
    ber_values = []

    # Generate random data bits
    bits = generate_bits(n_bits)

    # Modulate bits into QPSK symbols
    symbols = qpsk_modulate(bits)

    # Run simulation for different SNR values
    for snr_db in snr_db_values:

        # Pass symbols through AWGN channel
        received_symbols = add_awgn_noise(symbols, snr_db)

        # Demodulate received symbols
        received_bits = qpsk_demodulate(received_symbols)

        # Calculate BER
        ber = calculate_ber(bits, received_bits)

        ber_values.append(ber)

        print(f"SNR = {snr_db:2d} dB, BER = {ber:.6f}")

    # --------------------------------------------------------
    # Plot BER vs SNR
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Use logarithmic scale for BER
    plt.semilogy(snr_db_values, ber_values, marker="o")

    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")

    plt.title("QPSK BER Performance over AWGN Channel")

    plt.tight_layout()

    # Save plot as image
    plt.savefig("ber_plot.png", dpi=300)

    # Show plot
    plt.show()


# Run the simulation
if __name__ == "__main__":
    main()

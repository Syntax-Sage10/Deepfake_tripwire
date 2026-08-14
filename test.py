import os
import sys
import argparse
import librosa
import numpy as np


class ComprehensiveVoiceTripwire:
    # Centralized so they're easy to find and tune in one place.
    DEFAULT_THRESHOLDS = {
        "jitter_low": 0.003,
        "jitter_high": 0.09,
        "jitter_weight": 0.20,

        "shimmer_low": 0.008,
        "shimmer_weight": 0.15,

        "mfcc_delta_low": 12.0,
        "mfcc_delta_weight": 0.25,

        "phase_low": 50.0,
        "phase_high": 100000.0,
        "phase_weight": 0.20,

        "flatness_low": 0.00002,
        "flatness_high": 0.015,
        "flatness_weight": 0.15,

        "verdict_cutoff": 0.35,
    }

    def __init__(self, target_sr=16000, thresholds: dict = None):
        self.sr = target_sr
        # Allow overriding thresholds per-instance without editing the class.
        self.t = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def analyze(self, file_path: str, verbose: bool = True) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at path: {file_path}")

        # 1. Load audio wave x[n]
        y, sr = librosa.load(file_path, sr=self.sr)

        # Trim leading and trailing silences
        y, _ = librosa.effects.trim(y, top_db=25)

        # Peak normalize waveform to eliminate volume/mic-distance bias
        if np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)

        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration < 0.3:
            raise ValueError("Audio clip is too short for mathematical signal verification (minimum 0.3s required).")

        # ------------------------------------------------------------------
        # 2. TIME-FREQUENCY SPECTROGRAM COMPUTATION (STFT)
        # ------------------------------------------------------------------
        n_fft = 2048
        hop_length = 512
        stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        stft_mag = np.abs(stft_complex)
        stft_phase = np.angle(stft_complex)
        power_spec = stft_mag ** 2

        # ------------------------------------------------------------------
        # 3. FUNDAMENTAL FREQUENCY (F0), JITTER & SHIMMER ANALYSIS
        # ------------------------------------------------------------------
        # Pass hop_length explicitly so f0/voiced_flag frames line up 1:1
        # with the RMS energy frames computed below (same hop_length).
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=hop_length
        )
        voiced_flag = np.asarray(voiced_flag, dtype=bool) if voiced_flag is not None else np.array([], dtype=bool)
        valid_f0 = f0[voiced_flag] if f0 is not None else np.array([])
        # Drop any leftover NaNs defensively (pyin can leave NaN even on
        # frames marked voiced in rare edge cases).
        valid_f0 = valid_f0[~np.isnan(valid_f0)]

        if len(valid_f0) > 3:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
            periods = 1.0 / valid_f0
            jitter_local = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))

            # FIX: previously this just took the *first N* energy frames
            # (frame_energies[:len(valid_f0)]), which has nothing to do with
            # which frames were actually voiced - it silently measured
            # energy variation over arbitrary frames, often including
            # silence/unvoiced segments. Now we use the same boolean voiced
            # mask that was used for f0, aligned to the same hop_length, so
            # shimmer is computed over the same frames as jitter.
            frame_energies = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
            n = min(len(frame_energies), len(voiced_flag))
            aligned_mask = voiced_flag[:n]
            valid_energies = frame_energies[:n][aligned_mask]

            if len(valid_energies) > 3:
                shimmer_local = float(np.mean(np.abs(np.diff(valid_energies))) / (np.mean(valid_energies) + 1e-8))
            else:
                shimmer_local = 0.0

            f0_valid = True
        else:
            f0_mean, f0_std, jitter_local, shimmer_local = 0.0, 0.0, 0.0, 0.0
            f0_valid = False

        # ------------------------------------------------------------------
        # 4. HARMONIC-TO-NOISE RATIO (HNR)
        # ------------------------------------------------------------------
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        noise_energy = np.sum(y_percussive ** 2) + 1e-8
        hnr_db = float(10.0 * np.log10(harmonic_energy / noise_energy))

        # ------------------------------------------------------------------
        # 5. SPECTRAL FLATNESS & CENTROID
        # ------------------------------------------------------------------
        flatness = librosa.feature.spectral_flatness(S=stft_mag)[0]
        flatness_mean = float(np.mean(flatness))

        # ------------------------------------------------------------------
        # 6. HIGH-FREQUENCY ENERGY RATIO (HFER)
        # ------------------------------------------------------------------
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        high_freq_mask = fft_freqs >= 7000.0
        total_energy = np.sum(power_spec) + 1e-8
        high_freq_energy = np.sum(power_spec[high_freq_mask, :])
        hfer_ratio = float(high_freq_energy / total_energy)

        # ------------------------------------------------------------------
        # 7. MFCCs, VELOCITY (Δ), AND ACCELERATION (Δ²)
        # ------------------------------------------------------------------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs, order=1)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        delta_var_mean = float(np.mean(np.var(mfcc_delta, axis=1)))
        delta2_var_mean = float(np.mean(np.var(mfcc_delta2, axis=1)))

        # ------------------------------------------------------------------
        # 8. SPECTRAL ROLLOFF (85%)
        # ------------------------------------------------------------------
        rolloff_85 = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, roll_percent=0.85)[0]
        rolloff_85_mean = float(np.mean(rolloff_85))

        # ------------------------------------------------------------------
        # 9. SPECTRAL FLUX
        # ------------------------------------------------------------------
        spectral_flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
        flux_var = float(np.var(spectral_flux))

        # ------------------------------------------------------------------
        # 10. PHASE INSTABILITY ANALYSIS
        # ------------------------------------------------------------------
        phase_diff = np.diff(stft_phase, axis=1)
        phase_unwrapped = np.unwrap(phase_diff, axis=0)
        phase_instability_var = float(np.mean(np.var(phase_unwrapped, axis=1)))

        # ------------------------------------------------------------------
        # DISCRIMINATOR ENGINE
        # ------------------------------------------------------------------
        t = self.t
        anomaly_vector = {}

        if f0_valid and (jitter_local < t["jitter_low"] or jitter_local > t["jitter_high"]):
            anomaly_vector["jitter"] = t["jitter_weight"]

        if f0_valid and shimmer_local < t["shimmer_low"]:
            anomaly_vector["shimmer"] = t["shimmer_weight"]

        if delta_var_mean < t["mfcc_delta_low"]:
            anomaly_vector["mfcc_delta_smoothness"] = t["mfcc_delta_weight"]

        if phase_instability_var > t["phase_high"] or phase_instability_var < t["phase_low"]:
            anomaly_vector["phase_anomaly"] = t["phase_weight"]

        if flatness_mean > t["flatness_high"] or flatness_mean < t["flatness_low"]:
            anomaly_vector["spectral_flatness"] = t["flatness_weight"]

        total_anomaly_score = sum(anomaly_vector.values())

        if verbose:
            print("\n" + "=" * 60)
            print(f" DIAGNOSTIC RUN FOR: {os.path.basename(file_path)}")
            print("=" * 60)
            print(f" • F0 Voiced Frames Found   : {len(valid_f0)} (valid={f0_valid})")
            print(f" • Jitter (Local)           : {jitter_local:.6f}")
            print(f" • Shimmer (Local)          : {shimmer_local:.6f}")
            print(f" • MFCC Delta Variance      : {delta_var_mean:.2f}")
            print(f" • Phase Instability Var    : {phase_instability_var:.4f}")
            print(f" • Spectral Flatness        : {flatness_mean:.7f}")
            print(f" • Triggered Anomalies      : {list(anomaly_vector.keys())}")
            print(f" • Total Anomaly Score      : {total_anomaly_score:.2f}")
            print("=" * 60 + "\n")

        is_deepfake = total_anomaly_score >= t["verdict_cutoff"]

        if is_deepfake:
            confidence = round(min(99.9, max(65.0, 50.0 + (total_anomaly_score * 60.0))), 1)
        else:
            confidence = round(min(99.9, max(65.0, 100.0 - (total_anomaly_score * 100.0))), 1)

        return {
            "duration_seconds": round(duration, 2),
            "verdict": "RED_SPOOF" if is_deepfake else "GREEN_HUMAN",
            "confidence_percent": confidence,
            "total_anomaly_score": round(total_anomaly_score, 3),
            "detected_anomalies": list(anomaly_vector.keys()),
            "mathematical_metrics": {
                "1_local_jitter": f"{jitter_local:.6f}",
                "2_local_shimmer": f"{shimmer_local:.6f}",
                "3_hnr_db": f"{hnr_db:.2f} dB",
                "4_spectral_flatness": f"{flatness_mean:.7f}",
                "5_high_freq_energy_ratio": f"{hfer_ratio:.6f}",
                "6_mfcc_delta_variance": f"{delta_var_mean:.2f}",
                "7_mfcc_delta2_variance": f"{delta2_var_mean:.2f}",
                "8_spectral_rolloff_85_hz": f"{rolloff_85_mean:.1f} Hz",
                "9_spectral_flux_variance": f"{flux_var:.4f}",
                "10_phase_instability_variance": f"{phase_instability_var:.4f}",
                "fundamental_frequency_f0_mean_hz": f"{f0_mean:.1f} Hz",
            },
        }


def _print_comparison_table(rows):
    """rows: list of (label, file_path, metrics_dict)"""
    if not rows:
        print("No files analyzed.")
        return

    cols = list(rows[0][2].keys())
    label_w = max(len(r[0]) for r in rows) + 2
    file_w = max(len(os.path.basename(r[1])) for r in rows) + 2

    header = f"{'Label':<{label_w}}{'File':<{file_w}}" + "".join(f"{c:<28}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, path, metrics in rows:
        line = f"{label:<{label_w}}{os.path.basename(path):<{file_w}}"
        for c in cols:
            line += f"{str(metrics[c]):<28}"
        print(line)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ComprehensiveVoiceTripwire on files and/or compare known real vs. fake samples."
    )
    parser.add_argument("--real", nargs="*", default=[], help="Paths to known REAL human voice clips")
    parser.add_argument("--fake", nargs="*", default=[], help="Paths to known AI-generated voice clips")
    parser.add_argument("files", nargs="*", help="Files to analyze without a known label")
    args = parser.parse_args()

    if not args.real and not args.fake and not args.files:
        parser.print_help()
        sys.exit(0)

    analyzer = ComprehensiveVoiceTripwire()
    rows = []

    for path in args.real:
        try:
            result = analyzer.analyze(path, verbose=True)
            rows.append(("REAL", path, result["mathematical_metrics"]))
        except Exception as e:
            print(f"[!] Failed to analyze {path}: {e}")

    for path in args.fake:
        try:
            result = analyzer.analyze(path, verbose=True)
            rows.append(("FAKE", path, result["mathematical_metrics"]))
        except Exception as e:
            print(f"[!] Failed to analyze {path}: {e}")

    for path in args.files:
        try:
            result = analyzer.analyze(path, verbose=True)
            rows.append(("?", path, result["mathematical_metrics"]))
        except Exception as e:
            print(f"[!] Failed to analyze {path}: {e}")

    if rows:
        _print_comparison_table(rows)
        print(
            "Compare the REAL vs FAKE rows above metric by metric. Any metric where "
            "REAL and FAKE clips overlap heavily is not a useful discriminator at its "
            "current threshold - widen or drop it in ComprehensiveVoiceTripwire.DEFAULT_THRESHOLDS. "
            "Any metric that cleanly separates the two groups is worth tightening around "
            "the midpoint between the REAL and FAKE ranges you observe."
        )
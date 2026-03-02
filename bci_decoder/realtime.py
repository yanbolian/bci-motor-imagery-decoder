"""
Real-time BCI decoding simulation with sliding-window inference.

This module replicates the core loop that runs in deployed BCI systems:

    1. Neural data streams in from hardware in small chunks (e.g., 10 samples
       = 40 ms at 250 Hz) — matching USB or Bluetooth acquisition latency.

    2. A ring buffer retains the most recent `window_sec` seconds.

    3. Every `step_sec` seconds, the buffer contents are decoded.

    4. The decoded class label (rest / left / right) is output to the BCI
       application (e.g., a cursor, a click intent, a typing interface).

Latency considerations
----------------------
In Synchron's deployed system the relevant latencies are:
    - Signal acquisition → streamed packet latency (~20–40 ms)
    - Preprocessing + feature extraction (~1–5 ms on CPU)
    - Model inference (~0.1–1 ms for LDA; ~2–5 ms for EEGNet on CPU)
    - Total system latency: typically 50–100 ms

This simulation measures the inference + preprocessing latency only,
which should be well below 10 ms on modern hardware — demonstrating
real-time viability.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Optional


class RingBuffer:
    """
    Fixed-size circular buffer for streaming multichannel neural signal data.

    Maintains the most recent `capacity` samples per channel.  New samples
    overwrite the oldest entries without reallocating memory — a key
    requirement for low-latency real-time signal processing.

    Parameters
    ----------
    n_channels : int
    capacity : int
        Maximum number of samples to hold (= window length in samples).
    """

    def __init__(self, n_channels: int, capacity: int):
        self.n_channels = n_channels
        self.capacity   = capacity
        self._buf       = np.zeros((n_channels, capacity), dtype=np.float32)
        self._write_idx = 0
        self._n_written = 0

    def push(self, chunk: np.ndarray) -> None:
        """
        Write a chunk of samples into the buffer.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, n_new_samples)
        """
        n_new = chunk.shape[1]
        for i in range(n_new):
            self._buf[:, self._write_idx] = chunk[:, i]
            self._write_idx = (self._write_idx + 1) % self.capacity
        self._n_written += n_new

    def get_latest(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Return the most recent `n_samples` samples in chronological order.

        Returns None if fewer than `n_samples` have been written yet.
        """
        if self._n_written < n_samples:
            return None
        start = (self._write_idx - n_samples) % self.capacity
        if start + n_samples <= self.capacity:
            return self._buf[:, start:start + n_samples].copy()
        part1 = self._buf[:, start:]
        part2 = self._buf[:, :n_samples - part1.shape[1]]
        return np.concatenate([part1, part2], axis=1)

    @property
    def is_ready(self) -> bool:
        """True once the buffer has been filled at least once."""
        return self._n_written >= self.capacity


class RealTimeDecoder:
    """
    Simulated real-time BCI decoder with sliding-window inference.

    Parameters
    ----------
    decode_fn : callable
        Function signature: (window: ndarray[1, n_channels, n_samples]) → int
        Takes a single decode window (with batch dimension) and returns
        a class label (0 = rest, 1 = left, 2 = right).
    n_channels : int
    sfreq : int
        Sampling frequency in Hz.
    window_sec : float
        Length of the decoding window in seconds.
    step_sec : float
        Stride between consecutive decode attempts in seconds.
        step_sec = 0.04 → 40 ms update rate (25 Hz output).
    chunk_size : int
        Samples per streaming chunk (simulates hardware packet size).
        At sfreq=250, chunk_size=10 → 40 ms packets.
    """

    def __init__(
        self,
        decode_fn: Callable,
        n_channels: int,
        sfreq: int = 250,
        window_sec: float = 1.0,
        step_sec: float = 0.04,
        chunk_size: int = 10,
    ):
        self.decode_fn      = decode_fn
        self.sfreq          = sfreq
        self.window_samples = int(window_sec * sfreq)
        self.step_samples   = int(step_sec * sfreq)
        self.chunk_size     = chunk_size

        self.buffer = RingBuffer(n_channels, self.window_samples)
        self._samples_since_decode = 0
        self.decode_log: List[Dict] = []

    def process_chunk(self, chunk: np.ndarray) -> Optional[int]:
        """
        Feed a new chunk into the ring buffer and decode if a step has elapsed.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, chunk_size)

        Returns
        -------
        label : int or None
            Decoded class label, or None if the step threshold was not reached.
        """
        self.buffer.push(chunk)
        self._samples_since_decode += chunk.shape[1]

        if self._samples_since_decode >= self.step_samples and self.buffer.is_ready:
            self._samples_since_decode = 0
            window = self.buffer.get_latest(self.window_samples)

            t0 = time.perf_counter()
            label = self.decode_fn(window[np.newaxis])  # add batch dim
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self.decode_log.append({
                "label":      int(label),
                "latency_ms": latency_ms,
                "window_end": self.buffer._n_written,   # total samples seen so far
            })
            return int(label)
        return None

    def run_simulation(
        self,
        stream_data: np.ndarray,
        true_labels: np.ndarray = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the decoder over a pre-recorded signal stream.

        This is the offline equivalent of running the decoder against live
        hardware — the signal is streamed chunk-by-chunk to replicate the
        real-time packet arrival pattern.

        Parameters
        ----------
        stream_data : ndarray, shape (n_channels, total_samples)
            Concatenated trials streamed through the decoder.
        true_labels : ndarray, shape (total_samples,), optional
            Per-sample ground-truth labels for accuracy estimation.
        verbose : bool

        Returns
        -------
        dict with keys:
            predictions   : ndarray of decoded labels
            latencies_ms  : ndarray of per-decode latencies in ms
            n_decodes     : int
            mean_latency_ms, max_latency_ms : float
            accuracy      : float (only if true_labels is provided)
        """
        self.decode_log = []
        n_samples = stream_data.shape[1]

        for start in range(0, n_samples - self.chunk_size + 1, self.chunk_size):
            chunk = stream_data[:, start:start + self.chunk_size]
            self.process_chunk(chunk)

        predictions = np.array([e["label"]      for e in self.decode_log])
        latencies   = np.array([e["latency_ms"] for e in self.decode_log])

        results: Dict = {
            "predictions":    predictions,
            "latencies_ms":   latencies,
            "n_decodes":      len(predictions),
            "mean_latency_ms": float(np.mean(latencies)) if len(latencies) else 0.0,
            "max_latency_ms":  float(np.max(latencies))  if len(latencies) else 0.0,
        }

        if true_labels is not None and len(predictions) > 0:
            # Anchor each decode to its window *start* (= window_end - window_samples).
            # This avoids mislabeling windows whose end crosses a trial boundary
            # but whose content is still entirely within the previous trial.
            window_starts = [
                max(0, e["window_end"] - self.window_samples)
                for e in self.decode_log
            ]
            matched = [
                true_labels[min(t, len(true_labels) - 1)]
                for t in window_starts
            ]
            results["accuracy"] = float(np.mean(predictions == np.array(matched)))

        if verbose:
            print(f"\n  Real-time simulation summary")
            print(f"    Decodes performed : {results['n_decodes']}")
            print(f"    Mean latency      : {results['mean_latency_ms']:.2f} ms")
            print(f"    Max  latency      : {results['max_latency_ms']:.2f} ms")
            if "accuracy" in results:
                print(f"    Stream accuracy   : {results['accuracy']:.3f}")

        return results

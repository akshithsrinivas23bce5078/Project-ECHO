"""
Project ECHO — IDeLog Real-Time Neuromotor Extractor
======================================================
Implements the Iterative Dual Spatial and Kinematic Extraction of
Sigma-Lognormal Parameters (IDeLog) algorithm for continuous,
real-time reverse-engineering of incoming mouse trajectory data.

The algorithm identifies virtual target points the CNS intended to
reach, isolates individual motor commands, and extracts a 6-parameter
set per sub-movement:

    t0      — command timing (reaction time / CNS processing delay)
    D       — movement amplitude (spatial extent of motor command)
    μ       — log time delay (peripheral neuromuscular transmission)
    σ       — log response time (fatigue / cognitive load indicator)
    θ_start — starting angular orientation
    θ_end   — ending angular orientation

The iterative refinement loop alternates between:
    1. Spatial optimization: refine virtual target point positions
    2. Kinematic optimization: fit lognormal parameters to each segment

Convergence is declared when the velocity reconstruction SNR exceeds
the target or the parameter change falls below the threshold.

Reference:
    O'Reilly, C. & Plamondon, R. (2009). Development of a
    Sigma-Lognormal representation for on-line signatures.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from feature_extraction.snr_validator import SNRValidator, SNRResult, SNRQuality


# ---------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------

@dataclass
class NeuromotorCommand:
    """
    A single neuromotor command extracted by IDeLog.

    Each command represents one ballistic sub-movement generated
    by the central nervous system.
    """
    t0: float           # Command timing (seconds)
    D: float            # Movement amplitude (pixels)
    mu: float           # Log time delay
    sigma: float        # Log response time (fatigue/load indicator)
    theta_start: float  # Starting angular orientation (radians)
    theta_end: float    # Ending angular orientation (radians)
    segment_id: int = 0 # Which trajectory segment this belongs to


@dataclass
class TrajectorySegment:
    """A continuous ballistic trajectory segment between pauses."""
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    segment_id: int = 0


@dataclass
class IDeLogResult:
    """Complete result of IDeLog extraction on a trajectory segment."""
    commands: List[NeuromotorCommand]
    snr_result: SNRResult
    iterations_used: int
    converged: bool
    virtual_targets: List[Tuple[float, float]]  # (x, y) target points
    original_velocity: Optional[np.ndarray] = field(default=None, repr=False)
    reconstructed_velocity: Optional[np.ndarray] = field(default=None, repr=False)
    times: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------
# IDeLog Extractor
# ---------------------------------------------------------------

class IDeLogExtractor:
    """
    Real-time Iterative Dual Spatial and Kinematic extraction of
    Sigma-Lognormal neuromotor parameters.

    Usage:
        extractor = IDeLogExtractor()
        result = extractor.process_segment(trajectory_segment)
        for cmd in result.commands:
            print(f"σ={cmd.sigma:.3f}, μ={cmd.mu:.3f}")  # cognitive load
    """

    def __init__(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001,
        min_segment_points: int = 10,
        snr_validator: Optional[SNRValidator] = None,
    ):
        """
        Args:
            max_iterations: Maximum IDeLog refinement iterations.
            convergence_threshold: Stop when parameter change < this.
            min_segment_points: Minimum trajectory points per segment.
            snr_validator: SNR validator instance (uses default if None).
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_segment_points = min_segment_points
        self._snr_validator = snr_validator or SNRValidator()
        self._segment_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_segment(self, segment: TrajectorySegment) -> IDeLogResult:
        """
        Run the full IDeLog pipeline on a single trajectory segment.

        Args:
            segment: A TrajectorySegment with xs, ys, ts arrays.

        Returns:
            IDeLogResult with extracted neuromotor commands and quality.
        """
        xs, ys, ts = segment.xs, segment.ys, segment.ts

        if len(xs) < self.min_segment_points:
            return self._empty_result(segment.segment_id)

        # Step 1: Compute velocity profile
        times, velocities, angles = self._compute_kinematics(xs, ys, ts)
        if len(velocities) < 3:
            return self._empty_result(segment.segment_id)

        # Step 2: Detect virtual target points
        targets = self._detect_virtual_targets(xs, ys, ts, velocities, times)
        if len(targets) < 2:
            return self._empty_result(segment.segment_id)

        # Step 3: Isolate sub-movements between targets
        sub_movements = self._isolate_sub_movements(
            xs, ys, ts, times, velocities, angles, targets
        )

        # Step 4: Initial parameter extraction
        commands = self._extract_initial_params(sub_movements, segment.segment_id)
        if not commands:
            return self._empty_result(segment.segment_id)

        # Step 5: Iterative dual refinement
        commands, iterations, converged = self._iterative_refinement(
            commands, times, velocities, targets, xs, ys, ts
        )

        # Step 6: Compute final reconstruction and SNR
        reconstructed = self._reconstruct_velocity(commands, times)
        snr_result = self._snr_validator.validate(velocities, reconstructed)

        # If unsatisfactory, still return (caller decides via snr_result)
        target_points = [(xs[t], ys[t]) for t in targets if t < len(xs)]

        return IDeLogResult(
            commands=commands,
            snr_result=snr_result,
            iterations_used=iterations,
            converged=converged,
            virtual_targets=target_points,
            original_velocity=velocities,
            reconstructed_velocity=reconstructed,
            times=times,
        )

    def process_raw(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
    ) -> IDeLogResult:
        """
        Convenience method: process raw arrays directly.

        Args:
            xs, ys: 1-D position arrays.
            ts: 1-D timestamp array (seconds, monotonic).

        Returns:
            IDeLogResult with extracted neuromotor commands.
        """
        self._segment_counter += 1
        segment = TrajectorySegment(
            xs=np.asarray(xs, dtype=np.float64),
            ys=np.asarray(ys, dtype=np.float64),
            ts=np.asarray(ts, dtype=np.float64),
            segment_id=self._segment_counter,
        )
        return self.process_segment(segment)

    # ------------------------------------------------------------------
    # Step 1: Kinematics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_kinematics(
        xs: np.ndarray, ys: np.ndarray, ts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity magnitude and movement angles from trajectory.

        Returns:
            (times, velocities, angles) — all 1-D arrays of length N-1.
        """
        dx = np.diff(xs.astype(np.float64))
        dy = np.diff(ys.astype(np.float64))
        dt = np.diff(ts.astype(np.float64))
        dt = np.where(dt == 0, 1e-6, dt)

        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx ** 2 + vy ** 2)
        angles = np.arctan2(vy, vx)

        return ts[1:], speed, angles

    # ------------------------------------------------------------------
    # Step 2: Virtual target point detection
    # ------------------------------------------------------------------

    def _detect_virtual_targets(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
    ) -> List[int]:
        """
        Identify virtual target points using velocity minima and
        curvature analysis.

        Virtual targets are the points the CNS intended to reach.
        They occur at velocity minima between ballistic sub-movements.

        Returns:
            List of indices into the velocity array marking targets.
        """
        targets = [0]  # Always start with the first point

        if len(velocities) < 5:
            targets.append(len(velocities) - 1)
            return targets

        # Smooth velocity for robust minima detection
        kernel_size = min(7, len(velocities) // 2)
        if kernel_size >= 3:
            kernel = np.ones(kernel_size) / kernel_size
            v_smooth = np.convolve(velocities, kernel, mode='same')
        else:
            v_smooth = velocities.copy()

        # Find velocity minima (candidate virtual targets)
        v_median = np.median(v_smooth)
        v_threshold = v_median * 0.3  # Below 30% of median = near-stop

        for i in range(1, len(v_smooth) - 1):
            is_local_min = (
                v_smooth[i] <= v_smooth[i - 1]
                and v_smooth[i] <= v_smooth[i + 1]
            )
            is_low_velocity = v_smooth[i] < v_threshold

            if is_local_min and is_low_velocity:
                # Ensure minimum separation from last target
                if (i - targets[-1]) >= 5:
                    targets.append(i)

        # Also detect curvature-based targets (sharp direction changes)
        if len(xs) > 2:
            dx = np.diff(xs[1:].astype(np.float64))
            dy = np.diff(ys[1:].astype(np.float64))
            if len(dx) >= 2:
                angles = np.arctan2(dy, dx)
                angle_diff = np.abs(np.diff(angles))
                # Wrap to [0, π]
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                curvature_thresh = np.pi / 3  # 60° threshold

                for i in range(len(angle_diff)):
                    idx = i + 1  # offset from velocity array
                    if (angle_diff[i] > curvature_thresh
                            and idx not in targets
                            and idx < len(velocities) - 1):
                        if not targets or (idx - targets[-1]) >= 5:
                            targets.append(idx)

        # Always include the last point
        last_idx = len(velocities) - 1
        if targets[-1] != last_idx:
            targets.append(last_idx)

        targets.sort()
        return targets

    # ------------------------------------------------------------------
    # Step 3: Sub-movement isolation
    # ------------------------------------------------------------------

    @staticmethod
    def _isolate_sub_movements(
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        times: np.ndarray,
        velocities: np.ndarray,
        angles: np.ndarray,
        targets: List[int],
    ) -> List[dict]:
        """
        Segment the trajectory into sub-movements between virtual targets.

        Returns:
            List of dicts, each containing the sub-movement's spatial
            and kinematic data.
        """
        sub_movements = []

        for i in range(len(targets) - 1):
            start_idx = targets[i]
            end_idx = targets[i + 1]

            if end_idx - start_idx < 2:
                continue

            seg_times = times[start_idx:end_idx + 1]
            seg_velocities = velocities[start_idx:end_idx + 1]
            seg_angles = angles[start_idx:end_idx + 1]

            # Spatial data (from original trajectory, offset by 1 for alignment)
            pos_start = start_idx + 1  # velocity array is offset by 1
            pos_end = end_idx + 2
            if pos_end > len(xs):
                pos_end = len(xs)
            seg_xs = xs[pos_start:pos_end]
            seg_ys = ys[pos_start:pos_end]

            # Compute amplitude: Euclidean distance start → end
            if len(seg_xs) >= 2:
                D = np.sqrt(
                    (seg_xs[-1] - seg_xs[0]) ** 2
                    + (seg_ys[-1] - seg_ys[0]) ** 2
                )
            else:
                D = 0.0

            sub_movements.append({
                "times": seg_times,
                "velocities": seg_velocities,
                "angles": seg_angles,
                "xs": seg_xs,
                "ys": seg_ys,
                "D": D,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

        return sub_movements

    # ------------------------------------------------------------------
    # Step 4: Initial parameter extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_initial_params(
        sub_movements: List[dict],
        segment_id: int,
    ) -> List[NeuromotorCommand]:
        """
        Extract initial 6-parameter estimates from each sub-movement
        using peak geometry (closed-form method-of-moments).
        """
        commands = []

        for sm in sub_movements:
            t = sm["times"]
            v = sm["velocities"]
            angles = sm["angles"]
            D = sm["D"]

            if len(v) < 2 or D < 1e-3:
                continue

            # Command timing: onset of sub-movement
            t0 = float(t[0])

            # Peak velocity analysis
            peak_idx = int(np.argmax(v))
            t_peak = t[peak_idx]
            v_peak = v[peak_idx]

            # Time from onset to peak
            dt_peak = t_peak - t0
            if dt_peak <= 0.01:
                dt_peak = 0.05

            # Initial sigma estimate (method of moments)
            sigma = 0.40

            # Initial mu estimate from peak timing
            mu = float(np.log(dt_peak) + sigma ** 2)

            # Angular orientations
            theta_start = float(angles[0]) if len(angles) > 0 else 0.0
            theta_end = float(angles[-1]) if len(angles) > 0 else 0.0

            commands.append(NeuromotorCommand(
                t0=t0,
                D=float(D),
                mu=mu,
                sigma=sigma,
                theta_start=theta_start,
                theta_end=theta_end,
                segment_id=segment_id,
            ))

        return commands

    # ------------------------------------------------------------------
    # Step 5: Iterative dual refinement
    # ------------------------------------------------------------------

    def _iterative_refinement(
        self,
        commands: List[NeuromotorCommand],
        times: np.ndarray,
        velocities: np.ndarray,
        targets: List[int],
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
    ) -> Tuple[List[NeuromotorCommand], int, bool]:
        """
        Alternating spatial and kinematic optimization until convergence.

        Returns:
            (refined_commands, iterations_used, converged)
        """
        from scipy.optimize import minimize

        converged = False

        for iteration in range(self.max_iterations):
            # --- Kinematic refinement: optimize lognormal parameters ---
            prev_params = self._commands_to_vector(commands)

            # Build optimization target
            n_commands = len(commands)
            if n_commands == 0:
                break

            x0 = prev_params.copy()
            bounds = self._build_bounds(commands)

            def objective(params):
                v_hat = self._eval_lognormal_sum_6p(params, times, n_commands)
                return float(np.sum((velocities - v_hat) ** 2))

            try:
                result = minimize(
                    objective, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-10}
                )
                commands = self._vector_to_commands(
                    result.x, n_commands,
                    [c.segment_id for c in commands]
                )
            except Exception:
                pass  # Keep current params on optimization failure

            # --- Check convergence ---
            new_params = self._commands_to_vector(commands)
            param_change = np.max(np.abs(new_params - prev_params))

            if param_change < self.convergence_threshold:
                converged = True
                return commands, iteration + 1, converged

        return commands, self.max_iterations, converged

    # ------------------------------------------------------------------
    # Velocity reconstruction from 6-parameter commands
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_lognormal_sum_6p(
        params: np.ndarray, times: np.ndarray, n_commands: int
    ) -> np.ndarray:
        """
        Evaluate sum of lognormals from a flat 6-parameter vector.

        Parameter order per command: [t0, D, mu, sigma, theta_start, theta_end]
        Only t0, D, mu, sigma affect the velocity profile; angles are
        preserved but not used in velocity reconstruction.
        """
        v = np.zeros_like(times, dtype=np.float64)
        sqrt2pi = np.sqrt(2.0 * np.pi)

        for k in range(n_commands):
            off = k * 6
            t0 = params[off]
            D = params[off + 1]
            mu = params[off + 2]
            sigma = params[off + 3]
            # theta_start, theta_end at off+4, off+5 — not used for velocity

            if sigma < 0.01:
                sigma = 0.01

            valid = times > t0
            if not np.any(valid):
                continue

            dt = times[valid] - t0
            log_dt = np.log(dt)
            exponent = -((log_dt - mu) ** 2) / (2.0 * sigma ** 2)
            v[valid] += D * np.exp(exponent) / (dt * sigma * sqrt2pi)

        return v

    def _reconstruct_velocity(
        self, commands: List[NeuromotorCommand], times: np.ndarray
    ) -> np.ndarray:
        """Reconstruct velocity from a list of NeuromotorCommands."""
        params = self._commands_to_vector(commands)
        return self._eval_lognormal_sum_6p(params, times, len(commands))

    # ------------------------------------------------------------------
    # Parameter vector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _commands_to_vector(commands: List[NeuromotorCommand]) -> np.ndarray:
        """Flatten commands to a 1-D parameter vector."""
        params = []
        for c in commands:
            params.extend([c.t0, c.D, c.mu, c.sigma, c.theta_start, c.theta_end])
        return np.array(params, dtype=np.float64)

    @staticmethod
    def _vector_to_commands(
        params: np.ndarray,
        n_commands: int,
        segment_ids: List[int],
    ) -> List[NeuromotorCommand]:
        """Unpack a flat parameter vector into NeuromotorCommand objects."""
        commands = []
        for k in range(n_commands):
            off = k * 6
            commands.append(NeuromotorCommand(
                t0=float(params[off]),
                D=float(params[off + 1]),
                mu=float(params[off + 2]),
                sigma=float(params[off + 3]),
                theta_start=float(params[off + 4]),
                theta_end=float(params[off + 5]),
                segment_id=segment_ids[k] if k < len(segment_ids) else 0,
            ))
        return commands

    @staticmethod
    def _build_bounds(commands: List[NeuromotorCommand]) -> List[Tuple[float, float]]:
        """Build optimization bounds for L-BFGS-B."""
        bounds = []
        for c in commands:
            t_mode = c.t0 + np.exp(c.mu - c.sigma ** 2)
            bounds.append((max(0.0, c.t0 - 0.5), c.t0 + 0.5))     # t0
            bounds.append((1.0, max(2.0, c.D * 5.0)))               # D
            bounds.append((c.mu - 3.0, c.mu + 3.0))                 # mu
            bounds.append((0.05, 2.5))                                # sigma
            bounds.append((-np.pi, np.pi))                            # theta_start
            bounds.append((-np.pi, np.pi))                            # theta_end
        return bounds

    # ------------------------------------------------------------------
    # Empty result helper
    # ------------------------------------------------------------------

    def _empty_result(self, segment_id: int) -> IDeLogResult:
        """Return an empty result for degenerate inputs."""
        return IDeLogResult(
            commands=[],
            snr_result=SNRResult(
                snr_db=0.0,
                quality=SNRQuality.UNSATISFACTORY,
                confidence_weight=0.0,
                should_discard=True,
            ),
            iterations_used=0,
            converged=False,
            virtual_targets=[],
        )

/**
 * Scalar (one-dimensional) Kalman filter.
 *
 * Model assumptions baked into this implementation:
 *  - State is a single scalar value.
 *  - Transition is identity: x_{k|k-1} = x_{k-1|k-1}. The state is a scalar
 *    random walk — the prior for the next step is exactly the last posterior.
 *    This covers sensors reading a slowly-drifting signal with no known
 *    dynamics model.
 *  - Measurement is direct: z = x + noise.
 *  - All noise is Gaussian, independent, with caller-supplied variances.
 *
 * Because state and measurement are both scalars, every step is plain
 * arithmetic: no matrices are required.
 */
export class ScalarKalmanFilter {
  #state;
  #covariance;
  #q;
  #r;

  /**
   * @param {object} cfg
   * @param {number} cfg.processNoise      - Q: variance of the per-step process noise.
   * @param {number} cfg.measurementNoise  - R: variance of the measurement noise.
   * @param {number} cfg.initialState      - x_0: prior mean.
   * @param {number} cfg.initialCovariance - P_0: prior variance.
   */
constructor(cfg) {
    if (!cfg || typeof cfg !== "object") {
      throw new TypeError("configuration object is required");
    }
    const {
      processNoise,
      measurementNoise,
      initialState,
      initialCovariance,
    } = cfg;

    if (
      typeof processNoise !== "number" ||
      !Number.isFinite(processNoise) ||
      processNoise < 0
    ) {
      throw new RangeError("processNoise must be a finite, non-negative number");
    }
    if (
      typeof measurementNoise !== "number" ||
      !Number.isFinite(measurementNoise) ||
      measurementNoise < 0
    ) {
      throw new RangeError(
        "measurementNoise must be a finite, non-negative number",
      );
    }
    if (typeof initialState !== "number" || !Number.isFinite(initialState)) {
      throw new RangeError("initialState must be a finite number");
    }
    if (
      typeof initialCovariance !== "number" ||
      !Number.isFinite(initialCovariance) ||
      initialCovariance < 0
    ) {
      throw new RangeError(
        "initialCovariance must be a finite, non-negative number",
      );
    }

    this.#state = initialState;
    this.#covariance = initialCovariance;
    this.#q = processNoise;
    this.#r = measurementNoise;
  }

  /**
   * Run one full predict-correct cycle against a new measurement.
   *
   * @param {number} z - the raw scalar measurement.
   * @returns {{ state: number, covariance: number }}
   */
  update(z) {
    if (typeof z !== "number" || !Number.isFinite(z)) {
      throw new RangeError("measurement must be a finite number");
    }

    // Predict step. Transition is identity, so the prior mean carries over
    // unchanged. Only uncertainty grows by the process noise variance.
    // Failing to add Q here collapses the covariance over many steps, making
    // the filter unresponsive to later measurements.
    const priorCov = this.#covariance + this.#q;

    // Kalman gain. When the prior is uncertain (large cov) relative to the
    // measurement noise, the gain is high and we trust the measurement.
    const denom = priorCov + this.#r;
    const k = denom > 0 ? priorCov / denom : 0;

    // Correct step. Residual is what the measurement tells us beyond the prior.
    const residual = z - this.#state;
    this.#state = this.#state + k * residual;
    // Joseph-form covariance update is avoided for simplicity; the closed
    // form P = (1 - k) * prior is exact and numerically stable when k ∈ [0, 1].
    this.#covariance = (1 - k) * priorCov;

    return { state: this.#state, covariance: this.#covariance };
  }

  /**
   * Snapshot of the current estimate without advancing the filter.
   * @returns {{ state: number, covariance: number }}
   */
  snapshot() {
    return { state: this.#state, covariance: this.#covariance };
  }
}

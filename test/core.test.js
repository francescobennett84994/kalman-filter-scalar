import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { ScalarKalmanFilter } from "../src/index.js";

describe("ScalarKalmanFilter constructor", () => {
  it("accepts a valid configuration", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 0.1,
      measurementNoise: 1,
      initialState: 5,
      initialCovariance: 2,
    });
    const s = kf.snapshot();
    assert.strictEqual(s.state, 5);
    assert.strictEqual(s.covariance, 2);
  });

  it("throws on negative processNoise", () => {
    assert.throws(
      () =>
        new ScalarKalmanFilter({
          processNoise: -1,
          measurementNoise: 1,
          initialState: 0,
          initialCovariance: 1,
        }),
      RangeError,
    );
  });

  it("throws on negative measurementNoise", () => {
    assert.throws(
      () =>
        new ScalarKalmanFilter({
          processNoise: 1,
          measurementNoise: -1,
          initialState: 0,
          initialCovariance: 1,
        }),
      RangeError,
    );
  });

  it("throws on NaN initialState", () => {
    assert.throws(
      () =>
        new ScalarKalmanFilter({
          processNoise: 1,
          measurementNoise: 1,
          initialState: NaN,
          initialCovariance: 1,
        }),
      RangeError,
    );
  });

  it("throws on missing configuration object", () => {
    assert.throws(() => new ScalarKalmanFilter(), TypeError);
  });
});

describe("ScalarKalmanFilter.update", () => {
  it("throws on NaN measurement", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 1,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    assert.throws(() => kf.update(NaN), RangeError);
  });

  it("throws on non-finite measurement", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 1,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    assert.throws(() => kf.update(Infinity), RangeError);
  });

  it("returns the Kalman gain when Q=0 and R=0", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 0,
      measurementNoise: 0,
      initialState: 0,
      initialCovariance: 1,
    });
    const out = kf.update(5);
    // posterior = prior + 1 * (5 - 0) = 5
    assert.strictEqual(out.state, 5);
    assert.strictEqual(out.covariance, 0);
  });

  it("weights a noisy measurement with moderate gain", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 0.5,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    const out = kf.update(4);
    // prior cov = 1 + 0.5 = 1.5
    // gain = 1.5 / (1.5 + 1) = 0.6
    // state = 0 + 0.6 * 4 = 2.4
    assert.strictEqual(out.state, 2.4);
    // cov = (1 - 0.6) * 1.5 = 0.6
    assert.ok(Math.abs(out.covariance - 0.6) < 1e-9);
  });

  it("converges to the true mean over many iterations", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 1e-3,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    const truth = 10;
    for (let i = 0; i < 1000; i++) {
      kf.update(truth);
    }
    const final = kf.snapshot();
    assert.ok(final.state > 9.9 && final.state < 10.1);
    assert.ok(final.covariance < 1);
  });

  it("snapshots without advancing the filter", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 0.5,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    const snap = kf.snapshot();
    assert.strictEqual(snap.state, 0);
    assert.strictEqual(snap.covariance, 1);
    const out = kf.update(4);
    assert.strictEqual(out.state, 2.4);
  });

  it("tightening the covariance across multiple updates", () => {
    const kf = new ScalarKalmanFilter({
      processNoise: 0,
      measurementNoise: 1,
      initialState: 0,
      initialCovariance: 1,
    });
    const out1 = kf.update(5);
    assert.ok(out1.covariance < 1);
    assert.ok(out1.covariance > 0);
    const out2 = kf.update(5);
    assert.ok(out2.covariance < out1.covariance);
  });
});

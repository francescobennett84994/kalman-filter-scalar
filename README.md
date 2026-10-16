# kalman-filter-scalar

Recursively estimates the state of a noisy scalar process from a stream of noisy measurements.

## Usage

```js
import { ScalarKalmanFilter } from "kalman-filter-scalar";

const kf = new ScalarKalmanFilter({
  processNoise: 1e-4,
  measurementNoise: 0.01,
  initialState: 0,
  initialCovariance: 1,
});

const out1 = kf.update(2.5);
const out2 = kf.update(2.6);

console.log(out2.state); // the posterior estimate
console.log(out2.covariance); // the posterior covariance
```

Each `update(z)` runs a full predict-correct cycle and returns `{ state, covariance }`. The filter is stateful: subsequent calls build on the previous posterior.

## Why

Online signal smoothing without external state. The first measurement arrives, you get a posterior; the second arrives, you fold it in — no batching, no fixed window. The cost is one matrix-free scalar pass per sample.

This is a classical 1-D filter: `update()` performs the predict step (state is a scalar random walk so it carries over unchanged) then the correct step using the Kalman gain. Anything beyond scalar state — multi-dimensional observations, control inputs, non-identity transition matrices — is out of scope. Use a matrix-based library for that.

## Edge cases

A negative `processNoise` or `measurementNoise` will throw. A `NaN` or non-finite measurement will throw. `initialState` and `initialCovariance` may be any finite numbers; the first `update` call naturally incorporates the initial covariance through the gain. Providing exactly two values keeps the constructor unambiguous; partial config is rejected rather than defaulted to a value the caller may not have intended.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.


## License
MIT

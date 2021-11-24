import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class LMS:
    def __init__(self, Wt, step_size=0.5):
        self.Wt = np.squeeze(getattr(Wt, "A", Wt))
        self.step_size = step_size

    def est(self, X, y, mode="prediction"):

        if mode == "prediction":
            X = np.squeeze(getattr(X, "A", X))
            yest = self.Wt.dot(X)
            c = (y - yest) / X.dot(X)
            self.Wt += self.step_size * c * X

        elif mode == "cancelation":
            yest = self.Wt * X
            e = y - yest
            c = e.dot(e) / X.dot(X)
            self.Wt += self.step_size * c * X

        return yest


def chirp(n, f0=2, f1=40, t1=1):
    t = np.arange(n + 0.0) / n * t1

    return scipy.signal.chirp(t, f0, t1, f1)
    # return np.cos(2 * np.pi * f1 * t) + np.cos(4 * np.pi * f1 * t)


mode = "cancelation"
plot = 3

if __name__ == "__main__":
    for mode in ["prediction", "cancelation"]:
        if mode == "prediction":
            filter_len = 10
            step_size = 0.05
            signal_len = 2000
            f1 = 40
            noise_power = 0.05 * 2
            seed = 0

            np.random.seed(seed)

            signal_orig = chirp(signal_len, f1=f1)

            if noise_power:
                signal_orig += np.random.normal(scale=noise_power, size=signal_len)

            signal_orig *= 10

            targets = []
            yests = []

            lms = LMS(np.zeros(filter_len), step_size=step_size)

            for t in range(signal_len - filter_len):
                X = signal_orig[t: t + filter_len]
                target = signal_orig[t + filter_len]  # predict
                y_est = lms.est(X, target, mode=mode)
                targets += [target]
                yests += [y_est]

            target = np.array(targets)
            y_est = np.array(yests)
            error = y_est - target

            if plot:
                fig = plt.figure(figsize=(8, 7))
                plt.subplot(211)
                plt.plot(target, "k--", label="target")
                plt.title("Estimatied signal")
                plt.plot(y_est, label="predicted value")
                plt.legend()
                plt.grid(True)
                plt.subplot(212)
                plt.plot(error, label="error")
                plt.axhline(0, color="k", linestyle="--")
                plt.title("prediction error")
                plt.legend()
                plt.grid(True)

                if plot >= 2:
                    plt.savefig("../tmp_pred.png", dpi=300)
                plt.show()

        elif mode == "cancelation":
            filter_len = 10
            step_size = 0.1
            signal_len = 2000
            f1 = 40
            noise_power = 0.05 * 4
            seed = 0

            np.random.seed(seed)

            t = np.arange(signal_len + 0.0) / signal_len

            signal_orig = chirp(signal_len, f1=f1)

            if noise_power:
                noised_sig = (
                        signal_orig
                        + 0.05 * np.cos(2 * np.pi * (f1 / 4) * t)
                        + 0.2 * np.random.normal(scale=noise_power, size=signal_len)
                )

            signal_orig *= 10
            noised_sig *= 10

            targets = []
            yests = []
            iters = signal_len // filter_len
            lms = LMS(np.zeros(filter_len), step_size=step_size)

            for t in range(iters):
                X = noised_sig[filter_len * t: filter_len * (t + 1)]
                target = signal_orig[filter_len * t: filter_len * (t + 1)]
                y_est = lms.est(X, target, mode=mode)
                targets += [target]
                yests += [y_est]

            target = np.array(targets).flatten()
            y_est = np.array(yests).flatten()
            error = y_est - target

            if plot:
                fig = plt.figure(figsize=(8, 7))
                plt.subplot(211)
                plt.plot(noised_sig)
                plt.plot(y_est, label="predicted value")
                plt.plot(noised_sig, "r--", label="noised")
                plt.plot(signal_orig, "k--", label="target")
                plt.title("Cancelated signal")
                plt.legend(loc="lower left")
                plt.grid(True)
                plt.subplot(212)
                plt.plot(error, label="error")
                plt.axhline(0, color="k", linestyle="--")
                plt.title("prediction error")
                plt.legend(loc="lower left")
                plt.grid(True)

                if plot >= 2:
                    plt.savefig("../tmp_cancel.png", dpi=300)
                plt.show()

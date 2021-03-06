import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class AdaptiveFilterLMS:
    def __init__(self, filter_len=10, step_size=0.5):
        self.weights = np.zeros((filter_len))
        self.step_size = step_size

    def estimate(self, X, target, mode="prediction"):

        if mode == "prediction":
            y_estimated = self.weights.dot(X)
            del_J = (target - y_estimated) / X.dot(X)
            self.weights += self.step_size * del_J * X

        elif mode == "cancelation":
            y_estimated = self.weights * X
            error = target - y_estimated
            del_J = error.dot(error) / X.dot(X)
            self.weights += self.step_size * del_J * X

        return y_estimated


def get_chirp(n, f0=2, f1=40, t1=1):
    t = np.arange(n) / n * t1

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

            signal_orig = get_chirp(signal_len, f1=f1)

            if noise_power:
                signal_orig += np.random.normal(scale=noise_power, size=signal_len)

            signal_orig *= 10

            targets = []
            yests = []

            adaptivefilterlms = AdaptiveFilterLMS(
                filter_len=filter_len, step_size=step_size
            )

            for t in range(signal_len - filter_len):
                X = signal_orig[t : t + filter_len]
                target = signal_orig[t + filter_len]  # predict
                y_est = adaptivefilterlms.estimate(X, target, mode=mode)
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

            t = np.arange(signal_len) / signal_len
            signal_orig = get_chirp(signal_len, f1=f1)

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
            adaptivefilterlms = AdaptiveFilterLMS(
                filter_len=filter_len, step_size=step_size
            )

            for t in range(iters):
                X = noised_sig[filter_len * t : filter_len * (t + 1)]
                target = signal_orig[filter_len * t : filter_len * (t + 1)]
                y_est = adaptivefilterlms.estimate(X, target, mode=mode)
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

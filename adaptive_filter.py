import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class LMS:
    def __init__(self, Wt, damp=0.5):
        self.Wt = np.squeeze(getattr(Wt, "A", Wt))
        self.damp = damp

    def est(self, X, y, mode="prediction"):

        if mode == "prediction":
            X = np.squeeze(getattr(X, "A", X))
            yest = self.Wt.dot(X)
            c = (y - yest) / X.dot(X)
            self.Wt += self.damp * c * X

        elif mode == "cancelation":
            yest = self.Wt * X
            e = y - yest
            c = e.dot(e) / X.dot(X)
            self.Wt += self.damp * c * X

        return yest


def chirp(n, f0=2, f1=40, t1=1):
    # from $scipy/signal/waveforms.py
    t = np.arange(n + 0.0) / n * t1

    return scipy.signal.chirp(t, f0, t1, f1)
    # return np.cos(2 * np.pi * f1 * t) + np.cos(4 * np.pi * f1 * t)


mode = "cancelation"
plot = 3

if __name__ == "__main__":
    for mode in ["prediction", "cancelation"]:
        if mode == "prediction":
            filterlen = 10
            damp = 0.05
            nx = 2000
            f1 = 40  # chirp
            noise = 0.05 * 2  # * swing
            seed = 0

            np.random.seed(seed)

            Xlong = chirp(nx, f1=f1)
            # Xlong = np.cos( 2*np.pi * freq * np.arange(nx) )
            if noise:
                Xlong += np.random.normal(scale=noise, size=nx)  # laplace ...
            Xlong *= 10

            ys = []
            yests = []

            lms = LMS(np.zeros(filterlen), damp=damp)
            for t in range(nx - filterlen):
                X = Xlong[t : t + filterlen]
                y = Xlong[t + filterlen]  # predict
                yest = lms.est(X, y, mode=mode)
                ys += [y]
                yests += [yest]

            y = np.array(ys)
            yest = np.array(yests)
            err = yest - y

            if plot:
                fig = plt.figure(figsize=(8, 7))
                # fig.set_size_inches(12, 8)
                # fig.suptitle("Signal prediction", fontsize=12)
                plt.subplot(211)
                plt.plot(y, "k--", label="target")
                plt.title("Estimatied signal")
                plt.plot(yest, label="predicted value")
                plt.legend()
                plt.grid(True)
                plt.subplot(212)
                plt.plot(err, label="error")
                plt.axhline(0, color="k", linestyle="--")
                plt.title("prediction error")
                plt.legend()
                plt.grid(True)

                if plot >= 2:
                    plt.savefig("../tmp_pred.png", dpi=300)
                plt.show()

        elif mode == "cancelation":

            filterlen = 10
            damp = 0.1
            nx = 2000
            f1 = 40  # chirp
            noise = 0.05 * 4  # * swing
            seed = 0

            np.random.seed(seed)
            t = np.arange(nx + 0.0) / nx

            Xlong = chirp(nx, f1=f1)
            # Xlong = np.cos( 2*np.pi * freq * np.arange(nx) )

            if noise:
                Xlong_noise = (
                    Xlong
                    + 0.05 * np.cos(2 * np.pi * (f1 / 4) * t)
                    + 0.2 * np.random.normal(scale=noise, size=nx)
                )  # laplace ...

            Xlong *= 10
            Xlong_noise *= 10

            ys = []
            yests = []
            iters = nx // filterlen
            lms = LMS(np.zeros(filterlen), damp=damp)

            for t in range(iters):
                X = Xlong_noise[filterlen * t : filterlen * (t + 1)]
                y = Xlong[filterlen * t : filterlen * (t + 1)]
                # y = X_hf[filterlen * t : filterlen * (t + 1)]
                yest = lms.est(X, y, mode=mode)
                ys += [y]
                yests += [yest]

            y = np.array(ys).flatten()
            yest = np.array(yests).flatten()
            err = yest - y

            if plot:
                fig = plt.figure(figsize=(8, 7))
                # fig.set_size_inches(12, 8)
                # fig.suptitle("Signal prediction", fontsize=12)
                plt.subplot(211)
                # plt.plot(Xlong_noise)
                plt.plot(yest, label="predicted value")
                # plt.plot(Xlong_noise, "r--", label="noised")
                plt.plot(Xlong, "k--", label="target")
                plt.title("Cancelated signal")
                plt.legend(loc="lower left")
                plt.grid(True)
                plt.subplot(212)
                plt.plot(err, label="error")
                plt.axhline(0, color="k", linestyle="--")
                plt.title("prediction error")
                plt.legend(loc="lower left")
                plt.grid(True)

                if plot >= 2:
                    plt.savefig("../tmp_cancel.png", dpi=300)
                plt.show()

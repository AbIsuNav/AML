import numpy as np


class rvmclass(object):
    def __init__(self, Xs, targets, noise=None):
        self.X = Xs
        self.targets = targets
        self.noise = noise
        self.N = len(Xs)
        self.M = self.N + 1
        self.m_ = 1e-2 * np.ones([self.M])  # # mean posterior N
        self.sigma = np.zeros((self.M, self.M))  # covariance posterior N
        self.alpha = np.random.rand(self.M)
        self.phi = np.zeros([self.N, self.M])
        self.gamma = 0
        self.threshold_alpha = 1e9
        self.relevance_ = Xs
        self.tol = 1e-2
        self.beta = 1 / 0.0001
        self.iterations = 100000

    def Kernel(self):
        phi_ = np.zeros([self.N, self.M])
        for n in range(0, self.N):
            for m in range(0, self.M):
                if (m == 0):
                    phi_[n][m] = 1.0
                else:
                    phi_[n][m] = (np.minimum(self.X[n], self.X[m - 1]) ** 3) / 3 - (
                                np.minimum(self.X[n], self.X[m - 1]) ** 2) * (self.X[n] + self.X[m - 1]) / 2 + 1 + \
                                 self.X[n] * self.X[m - 1] + self.X[n]*self.X[m-1]*np.minimum(self.X[n], self.X[m-1])
        return phi_

    def Kernel_for_predict(self,relevance):
        phi_ = np.zeros([self.N, len(relevance)])
        for n in range(0, self.N):
            for m in range(0, len(relevance)):
                if (m == 0):
                    phi_[n][m] = 1.0
                else:
                    phi_[n][m] = (np.minimum(self.X[n], relevance[m - 1]) ** 3) / 3 - (
                                np.minimum(self.X[n], relevance[m - 1]) ** 2) * (self.X[n] + relevance[m - 1]) / 2 + 1 + \
                                 self.X[n] * relevance[m - 1] + self.X[n]*relevance[m-1]*np.minimum(self.X[n], relevance[m-1])
        return phi_


    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha < self.threshold_alpha

        keep_alpha[0] = True
        self.relevance_ = self.relevance_[keep_alpha[1:]]
        self.alpha = self.alpha[keep_alpha]
        self.phi = self.phi[:, keep_alpha]
        self.sigma = self.sigma[np.ix_(keep_alpha, keep_alpha)]
        self.m_ = self.m_[keep_alpha]
        print(keep_alpha.shape)
        return keep_alpha

    def predict(self, eval_MSE = False):
        """Evaluate the RVR model at x."""
        # phi = self.Kernel_for_predict(X)

        y = np.matmul(self.phi, self.m_)

        if eval_MSE:
            MSE = (1/self.beta) + np.matmul(self.phi, np.matmul(self.sigma, self.phi.T))
            return y, MSE[:, 0]
        else:
            return y

    def rvm(self):
        counter = 0
        self.phi = self.Kernel()
        alpha_old = self.alpha

        while self.iterations > counter:  # iterate until convergence
            counter += 1
            self.sigma = np.linalg.inv(np.diag(self.alpha) + self.beta*np.matmul(self.phi.T,self.phi))
            self.m_ = self.beta*np.matmul(self.sigma,np.matmul(self.phi.T,self.targets))
            self.gamma = 1 - self.alpha * np.diagonal(self.sigma)
            self.alpha = self.gamma / (self.m_ ** 2)
            self.beta = (self.N - np.sum(self.gamma)) / (np.linalg.norm((self.targets - np.matmul(self.phi, self.m_))) ** 2)
            keep_alpha = self._prune()
            delta = np.sum(np.absolute(self.alpha - alpha_old[keep_alpha]))
            print('Iteration: ', counter, 'Delta: ', delta)

            if delta < self.tol:
                break
            alpha_old = self.alpha
        print("Predicted sigma: ",(1/self.beta)**(1/2))
        return self.alpha[1:]


def create_sinc():
    x = np.linspace(-10.0, 10.0, num=100)
    sinc = np.zeros(len(x))
    for i in range(len(x)):
        sinc[i] = np.abs(x[i]) ** (-1) * np.sin(np.abs(x[i])) + np.random.normal(0,0.2)
    return x, sinc


if __name__ == '__main__':
    import matplotlib.pyplot as pl

    ax, ay = create_sinc()  # create data
    rv = rvmclass(ax, ay)
    alphas = rv.rvm()
    y_predicted = rv.predict()
    errors = np.abs(ay - y_predicted)
    y_original = np.abs(ax) ** (-1) * np.sin(np.abs(ax))
    print("Alphas: \n","(",len(alphas),")", alphas)
    print("---------------")
    print("max error: ", max(errors))
    print("min error: ", min(errors))

    x_new = rv.relevance_
    y_new = []
    for i in range(len(ax)):
        if ax[i] in x_new:
            y_new.append(y_predicted[i])

    pl.plot(ax, y_original)
    pl.scatter(ax, ay, marker='.')
    pl.scatter(x_new, y_new, facecolors='none', edgecolors='b')
    pl.show()

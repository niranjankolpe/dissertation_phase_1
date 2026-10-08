"""Smallest feature-token Transformer for tabular regression (numpy, hand-written backprop).
Each input value -> one d-dim token; 1 encoder layer (1-head self-attention + small feed-forward, both with residual);
mean-pool over tokens -> linear output. No layer-norm/dropout. Inputs and target are standardised internally."""
import numpy as np

class TinyTransformer:
    def __init__(self, seed=0, d=16, h=32, epochs=100, bs=512, lr=3e-3):
        self.seed, self.d, self.h, self.ep, self.bs, self.lr = seed, d, h, epochs, bs, lr

    def _init(self, F):
        r, d, h = np.random.default_rng(self.seed), self.d, self.h
        g = lambda *s: r.normal(0, 1 / np.sqrt(s[0] if len(s) == 2 else d), s).astype(np.float32)
        self.p = dict(Wt=r.normal(0, .5, (F, d)).astype(np.float32), bt=np.zeros((F, d), np.float32),
                      Wq=g(d, d), Wk=g(d, d), Wv=g(d, d), Wo=g(d, d) * .5, W1=g(d, h), b1=np.zeros(h, np.float32),
                      W2=g(h, d) * .5, b2=np.zeros(d, np.float32), wh=g(d, 1)[:, 0], bh=np.zeros((), np.float32))

    def _fwd(self, X):
        p, d = self.p, self.d
        T = X[:, :, None] * p["Wt"] + p["bt"]
        Q, K, V = T @ p["Wq"], T @ p["Wk"], T @ p["Wv"]
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(d); S = S - S.max(-1, keepdims=True)
        A = np.exp(S); A /= A.sum(-1, keepdims=True)
        O = A @ V; H = T + O @ p["Wo"]
        pre = H @ p["W1"] + p["b1"]; Z = np.maximum(pre, 0); Hf = H + Z @ p["W2"] + p["b2"]
        m = Hf.mean(1); return m @ p["wh"] + p["bh"], (X, T, Q, K, V, A, O, H, pre, Z, m)

    def _grad(self, X, t):
        p, d = self.p, self.d; y, (X, T, Q, K, V, A, O, H, pre, Z, m) = self._fwd(X); B, F = X.shape
        dy = 2 * (y - t) / B; g = {"wh": m.T @ dy, "bh": dy.sum()}
        dHf = (dy[:, None] * p["wh"])[:, None, :] / F * np.ones((B, F, d), np.float32)
        f2 = lambda a: a.reshape(-1, a.shape[-1])
        g["W2"], g["b2"] = f2(Z).T @ f2(dHf), dHf.sum((0, 1))
        dpre = (dHf @ p["W2"].T) * (pre > 0); g["W1"], g["b1"] = f2(H).T @ f2(dpre), dpre.sum((0, 1))
        dH = dHf + dpre @ p["W1"].T
        dO = dH @ p["Wo"].T; g["Wo"] = f2(O).T @ f2(dH)
        dA = dO @ V.transpose(0, 2, 1); dV = A.transpose(0, 2, 1) @ dO
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(d)
        dQ, dK = dS @ K, dS.transpose(0, 2, 1) @ Q
        g["Wq"], g["Wk"], g["Wv"] = f2(T).T @ f2(dQ), f2(T).T @ f2(dK), f2(T).T @ f2(dV)
        dT = dH + dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
        g["Wt"], g["bt"] = (dT * X[:, :, None]).sum(0), dT.sum(0)
        return g, float(((y - t) ** 2).mean())

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        self.mx, self.sx, self.my, self.sy = X.mean(0), X.std(0) + 1e-6, y.mean(), y.std() + 1e-6
        X, y = (X - self.mx) / self.sx, (y - self.my) / self.sy
        self._init(X.shape[1]); r = np.random.default_rng(self.seed + 1)
        m1 = {k: np.zeros_like(v) for k, v in self.p.items()}; m2 = {k: np.zeros_like(v) for k, v in self.p.items()}; k = 0
        nb = int(np.ceil(len(X) / self.bs)); tot = self.ep * nb
        for e in range(self.ep):
            idx = r.permutation(len(X))
            for b in range(nb):
                j = idx[b * self.bs:(b + 1) * self.bs]; g, _ = self._grad(X[j], y[j]); k += 1
                lr = self.lr * (1 - (k - 1) / tot) ** 1 if False else self.lr * .5 * (1 + np.cos(np.pi * k / tot))
                for n in self.p:
                    m1[n] = .9 * m1[n] + .1 * g[n]; m2[n] = .999 * m2[n] + .001 * g[n] ** 2
                    self.p[n] -= (lr * (m1[n] / (1 - .9 ** k)) / (np.sqrt(m2[n] / (1 - .999 ** k)) + 1e-8)).astype(np.float32)
        return self

    def predict(self, X):
        X = (np.asarray(X, np.float32) - self.mx) / self.sx
        return np.concatenate([self._fwd(X[i:i + 4096])[0] for i in range(0, len(X), 4096)]) * self.sy + self.my

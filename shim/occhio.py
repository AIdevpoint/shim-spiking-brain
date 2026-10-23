from __future__ import annotations

import math

import numpy as np

OCCHIO_PA = 60.0


def _griglia(n_punti, aspetto):
    gh = max(1, int(round(math.sqrt(n_punti / aspetto))))
    gw = max(1, n_punti // gh)
    return gw, gh


def _dog(img, s1, s2):
    def sfoca(x, s):
        r = max(1, int(3 * s))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        k /= k.sum()
        x = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, x)
        return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, x)
    return sfoca(img, s1) - sfoca(img, s2)


class Occhio:

    def __init__(self, c):
        self.lati = {}
        tutti = {"sx": c.gruppo("v1_dx"), "dx": c.gruppo("v1_sx")}

        self.gh = min(_griglia(len(v) // 2, 0.75)[1] for v in tutti.values())
        for lato, neuroni in tutti.items():
            gw = max(1, (len(neuroni) // 2) // self.gh)
            self.lati[lato] = {"neuroni": neuroni[:2 * gw * self.gh], "gw": gw, "gh": self.gh}
        self.gw = self.lati["sx"]["gw"] + self.lati["dx"]["gw"]
        self.neuroni = np.concatenate([self.lati["sx"]["neuroni"], self.lati["dx"]["neuroni"]])
        self.riposo = None

    def risoluzione(self):
        return self.gw, self.gh

    def correnti(self, img):
        img = img.astype(np.float32) / 255.0
        H, W = img.shape
        out = []
        self.contrasto_grezzo = 0.0
        for lato, x0, x1 in (("sx", 0, W // 2), ("dx", W // 2, W)):
            L = self.lati[lato]
            meta = img[:, x0:x1]

            ys = (np.arange(L["gh"]) + 0.5) * meta.shape[0] / L["gh"]
            xs = (np.arange(L["gw"]) + 0.5) * meta.shape[1] / L["gw"]
            passo = max(meta.shape[0] / L["gh"], 1.0)
            d = _dog(meta, 0.6 * passo, 1.8 * passo)
            g = d[np.clip(ys.astype(int), 0, meta.shape[0] - 1)][:, np.clip(xs.astype(int), 0, meta.shape[1] - 1)]
            self.contrasto_grezzo += float(np.abs(g).sum())
            g = g / (np.std(g) + 0.02)
            on, off = np.clip(g, 0, 3) / 3, np.clip(-g, 0, 3) / 3
            out.append(np.stack([on, off], -1).reshape(-1))
        return (np.concatenate(out) * OCCHIO_PA).astype(np.float32)

    def leggi(self, conta_v1, finestra_s):
        hz = conta_v1.astype(np.float32) / finestra_s
        if self.riposo is None:
            self.riposo = hz.copy()
        else:
            self.riposo += 0.01 * (hz - self.riposo)
        att = hz - self.riposo
        parti = []
        i = 0
        for lato in ("sx", "dx"):
            L = self.lati[lato]
            n = 2 * L["gw"] * L["gh"]
            a = att[i:i + n].reshape(L["gh"], L["gw"], 2)
            i += n
            parti.append(a[..., 0] - a[..., 1])
        im = np.concatenate(parti, 1)


        self.energia = float(np.abs(im).sum())
        return np.clip(im / (np.abs(im).max() + 1e-6), -1, 1)

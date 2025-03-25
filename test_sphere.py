import numpy as np
from BayHunter.surfdisp96_ext import surfdisp96
from BayHunter.surfdisp96_ext import sphere

# -----------------------------
# Étape 1 : Génération du modèle
# -----------------------------
nlayers = 6

z_bounds = [(100, 3000), (3000, 6000), (6000, 9000),
            (9000, 12000), (12000, 15000), (0, 0)]
vs_bounds = [(500, 4500), (500, 4500), (1000, 5000),
             (1000, 5000), (1000, 5000), (1000, 5000)]

np.random.seed(42)
z = np.array([np.random.uniform(*b) for b in z_bounds]) / 1000  # km
vs = np.array([np.random.uniform(*b) for b in vs_bounds]) / 1000  # km/s
vp = 1.73 * vs
rho = 0.31 * vs**0.25

# -----------------------------
# Étape 2 : Préparation buffers à la BayHunter
# -----------------------------
NL = 100
thk = np.zeros(NL)
vpm = np.zeros(NL)
vsm = np.zeros(NL)
rhom = np.zeros(NL)
rtp = np.zeros(NL)
dtp = np.zeros(NL)
btp = np.zeros(NL)

thk[:nlayers] = z
vpm[:nlayers] = vp
vsm[:nlayers] = vs
rhom[:nlayers] = rho

# -----------------------------
# Étape 3 : Appel à sphere (si elle est accessible comme surfdisp96)
# -----------------------------
ifunc = 2  # Rayleigh
iflag = 0  # Init
mmax = nlayers
llw = 0
two = 2 * np.pi

# sphere(ifunc, iflag, d, a, b, rho, rtp, dtp, btp, mmax, llw, two)
error = sphere(
    ifunc, iflag,
    thk, vpm, vsm, rhom,
    rtp, dtp, btp,
    mmax, llw, two
)

# -----------------------------
# Étape 4 : Affichage
# -----------------------------
print("=== Modèle initial ===")
for i in range(nlayers):
    print(f"Layer {i+1}: z={z[i]:.4f} km, vp={vp[i]:.4f}, vs={vs[i]:.4f}, rho={rho[i]:.4f}")

print("\n=== Résultat sphere ===")
for i in range(nlayers):
    print(f"Layer {i+1}: d={thk[i]:.4f}, a={vpm[i]:.4f}, b={vsm[i]:.4f}, "
          f"rho={rhom[i]:.4f}, btp={btp[i]:.4f}")

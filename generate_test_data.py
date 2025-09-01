# generate_test_data.py

# Imports
import numpy as np

# Initialisation
np.random.seed(42)

# Hyperparameters
nlayers = np.random.randint(5, 10)  # Layers
vpvs = np.random.uniform(1.5, 2.1) # Vp/Vs

# Synthetic model
thk = np.random.uniform(1, 10, size=nlayers)       # Thickness (km)
vs = np.random.uniform(2.0, 5.0, size=nlayers)      # Vs (km/s)
vp = vs * vpvs                                     # Vp (km/s)
rho = 0.32 * vp + 0.77                             # Approximation of density (g/cm^3)

# Convert
thk = thk.astype(np.float64)
vs = vs.astype(np.float64)
vp = vp.astype(np.float64)
rho = rho.astype(np.float64)

# Save
np.savez(
    "test_model_input.npz",
    thk=thk,
    vs=vs,
    vp=vp,
    rho=rho,
    nlayers=nlayers,
    vpvs=vpvs
)

print(f"Synthetic model saved with {nlayers} layers.")




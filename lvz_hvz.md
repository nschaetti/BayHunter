# LVZ and HVZ constraints in BayHunter

## What the parameters mean
- BayHunter lets you optionally suppress low‑velocity zones (LVZ) and high‑velocity zones (HVZ) when describing the shear‑wave velocity (`Vs`) profile with depth. Setting `lvz` or `hvz` to `None` keeps the prior uninformative, but supplying a positive fraction constrains how much `Vs` is allowed to drop or rise from one layer to the next (`docs/_sources/tutorial.rst.txt:244`‑`259`).
- The value is interpreted as a percentage. For example, `lvz = 0.1` allows at most a 10 % decrease of `Vs` from a layer to the layer immediately beneath it; `hvz = 0.1` would allow at most a 10 % increase (`docs/_sources/tutorial.rst.txt:248`‑`255`).
- Because `Vs` commonly increases with depth, clamping HVZ is only recommended when unrealistically high spikes appear; setting the threshold too low would smooth real discontinuities (`docs/_sources/tutorial.rst.txt:253`‑`257`).

## How the constraints are enforced while sampling
- During MCMC sampling each proposed model is first sorted, then validated before any forward modeling (`BayHunter/SingleChain.py:330`‑`407`). Validation checks the prior bounds (layer count, thickness, `Vs`, interface depths) and finally applies the LVZ/HVZ rules.
- The LVZ rule compares every pair of adjacent layers. A proposal passes only if `Vs_lower ≥ Vs_upper × (1 − lvz)`; equivalently, the computed vector `vs[1:] − vs[:-1] × (1 − lvz)` must stay positive, otherwise the model is rejected (`BayHunter/SingleChain.py:389`‑`397`).
- The HVZ rule mirrors the logic but limits increases: `Vs_lower ≤ Vs_upper × (1 + hvz)` or `(vs[:-1] × (1 + hvz)) − vs[1:] > 0`. Violations trigger an immediate rejection of the candidate model (`BayHunter/SingleChain.py:398`‑`405`).
- The same checks appear in the generic model validator used by dataset utilities (`BayHunter/data/validation.py:61`‑`99`), ensuring any model drawn from a `SeismicPrior` obeys LVZ/HVZ before being accepted into the sample set.
- Together these constraints act as hard priors: they do not alter the proposal distribution directly, but they filter out candidate velocity‑depth models that would introduce stronger drops or jumps than the user‑specified percentages. This keeps the Markov chains exploring only the physically plausible portion of the prior space.

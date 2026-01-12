# BayHunter — Architecture interne, fondements théoriques et cadre de comparaison avec MIGRATE

## 1. Positionnement général

**BayHunter** est un framework d’inversion bayésienne classique pour la sismologie, basé sur des **chaînes de Markov Monte Carlo (MCMC)**, incluant des mécanismes **trans-dimensionnels** (nombre de couches variable).
Il vise à **échantillonner le posterior exact** (au sens bayésien) défini par :

$$
p(\theta \mid d) \propto p(d \mid \theta)\, p(\theta)
$$

où :

* $\theta$ décrit un **modèle sismique 1D** (vitesses, profondeurs, nombre de couches, etc.),
* $d$ sont des observations (courbes de dispersion, receiver functions, …),
* le *forward model* (Surf96, rfmini, …) est supposé **fidèle et exact**.

Dans **MIGRATE**, BayHunter joue le rôle de **méthode de référence MCMC**, face à laquelle on compare :

* des **approximate posteriors** appris par des modèles génératifs,
* sous **contrainte explicite de budget de simulations**.

---

## 2. Architecture logicielle — vue d’ensemble

### 2.1 CLI et orchestration (point d’entrée)

**`BayHunter/__main__.py`**  
Fournit une **CLI orientée MIGRATE**, basée sur Click :

* génération de datasets,
* perturbations vectorisées,
* uploads Hugging Face,
* outils de visualisation.

Un point important pour MIGRATE est l’usage de **helpers de sûreté** (`safe_forward`) qui évitent qu’un forward pathologique bloque une chaîne complète.  
Cela reflète déjà une hypothèse implicite : *le forward model est coûteux et fragile*.

---

### 2.2 Orchestrateur MCMC multi-chaînes

**`BayHunter/mcmcOptimizer.py` — `MCMC_Optimizer`**

Rôle :

* charger la configuration et les priors,
* allouer des **buffers partagés** (modèles, misfits, bruit),
* instancier plusieurs chaînes indépendantes (`SingleChain`),
* gérer le **multiprocessing**,
* éventuellement diffuser l’état vers **BayWatch** (ZeroMQ).

Point clé pour MIGRATE :

> BayHunter est **fondamentalement multi-chaînes**, mais chaque chaîne reste **séquentielle** et **simulation-bound**.

La fonction centrale `mp_inversion` :

* contrôle le parallélisme,
* ne change **en rien** la complexité asymptotique : le coût reste proportionnel au **nombre total d’appels au forward**.

---

### 2.3 Logique MCMC par chaîne

**`BayHunter/SingleChain.py`**

Chaque chaîne implémente un **MCMC Metropolis–Hastings trans-dimensionnel** :

1. **Initialisation**
   * tirage depuis les priors,
   * validation des contraintes géophysiques.

2. **Boucle principale**
   * proposition (birth / death / move / perturbation),
   * calcul du forward,
   * évaluation de la vraisemblance,
   * acceptation / rejet.

3. **Adaptation**
   * ajustement dynamique des largeurs de propositions.

4. **Stockage**
   * burn-in vs phase principale,
   * sauvegarde des modèles acceptés, des log-likelihoods, etc.

👉 D’un point de vue théorique, BayHunter approxime le posterior via :

$$
\{\theta^{(t)}\}_{t=1}^T \sim p(\theta \mid d)
$$

sous réserve :

* de convergence,
* de mixing suffisant,
* d’un nombre de simulations **très élevé**.

---

## 3. Couplage données ↔ forward ↔ vraisemblance

### 3.1 Targets et forward modeling

**`BayHunter/Targets.py`**

Ce module formalise la séparation essentielle :

* **ObservedData** : données observées,
* **ModeledData** : réponses synthétiques,
* **Valuation** : vraisemblance et misfit.

Chaque *Target* (Rayleigh, Love, RF, etc.) :

* encapsule un plugin de forward modeling,
* fournit une vraisemblance spécifique (souvent gaussienne).

**`JointTarget.evaluate`** :

* appelle successivement les forward models,
* agrège les log-likelihoods,
* constitue **le goulet d’étranglement computationnel**.

> Pour MIGRATE, c’est **le point exact** où l’on doit instrumenter le comptage des simulations.

---

### 3.2 Hypothèses implicites fortes

BayHunter suppose :

* forward **exact** (pas d’erreur de modèle),
* bruit **bien spécifié**,
* indépendance conditionnelle entre datasets (souvent),
* stationnarité.

Ces hypothèses sont **rarement vérifiées** en pratique, mais intégrées *by design*.

---

## 4. Représentation des modèles sismiques

**`BayHunter/Models.py`**

Fonctions principales :

* conversion vecteurs ↔ couches,
* calcul de profils $v_s(z)$, $v_p(z)$,
* statistiques sur ensembles de modèles.

Important pour MIGRATE :

* BayHunter travaille dans un **espace de modèles explicite**, interprétable,
* MIGRATE travaille dans un **espace latent appris**, qui doit être validé *a posteriori*.

---

## 5. Génération de données synthétiques et datasets

Le sous-package **`BayHunter/data`** modernise BayHunter pour des usages type MIGRATE :

* **`SeismicPrior` / `SeismicParams`** : priors explicites, sérialisables,
* **`SeismicModel`** : représentation standardisée,
* **`DispersionCurve`** : abstraction propre des observables,
* **conversion Voronoï → couches** : clé pour compatibilité SBI,
* export **Arrow / Parquet / Hugging Face**.

👉 C’est **le pont conceptuel** entre BayHunter (MCMC) et MIGRATE (apprentissage amorti).

---

## 6. Visualisation et diagnostic

### 6.1 Post-hoc plotting

**`Plotting.py`**

* agrégation multi-chaînes,
* thinning,
* rejet d’outliers,
* figures finales (PDF).

### 6.2 Monitoring en ligne

**`BayWatch.py`**

* visualisation live des trajectoires,
* utile pour le debug,
* **pas** une métrique scientifique de convergence.

---

## 7. Comptage des simulations — point clé pour MIGRATE

### 7.1 Pourquoi c’est crucial

Pour comparer BayHunter à MIGRATE :

* le **temps mur** est trompeur,
* le **nombre d’itérations** est non comparable,
* **le seul budget comparable est le nombre de forwards**.

### 7.2 Point d’instrumentation correct

Tous les forwards passent par :

```text
SingleChain.iterate
 └── JointTarget.evaluate
     └── target.moddata.calc_synth
````

Donc :

* incrément **exactement au moment du forward**,
* indépendamment de l’acceptation.

### 7.3 Implication méthodologique

Cela permet :

* des courbes **misfit vs #simulations**,
* des **PPC vs budget**,
* une comparaison honnête avec un modèle génératif entraîné sur $N$ simulations.

---

## 8. Cadre de comparaison BayHunter ↔ MIGRATE

### 8.1 Ce qui est comparable

✔ Distributions a posteriori sur :

* paramètres dérivés,
* observables physiques,
* misfits prédictifs.

✔ PPC, coverage, rank histograms.

✔ Performance **à budget de simulations fixé**.

---

### 8.2 Ce qui ne l’est pas directement

✘ “BayHunter est exact, MIGRATE est approximatif”
→ faux sans analyse de convergence.

✘ Comparer une chaîne MCMC de $10^6$ forwards à un NPE entraîné sur $10^4$ simulations.

✘ Comparer MAP BayHunter à moyenne MIGRATE sans PPC.

---

## 9. Lecture critique (volontairement sceptique)

* BayHunter **n’est pas une vérité absolue**, mais :

  * une **approximation asymptotique**,
  * extrêmement coûteuse,
  * fragile hors hypothèses idéales.

* MIGRATE **n’essaie pas de battre BayHunter**, mais :

  * d’atteindre une **qualité statistique comparable**,
  * avec **2–3 ordres de grandeur de simulations en moins**,
  * et une généralisation amortie.

👉 La bonne question scientifique n’est pas :

> *“Est-ce que MIGRATE est aussi bon que BayHunter ?”*

mais :

> *“À budget de simulations fixé, quel est le meilleur estimateur prédictif et probabiliste ?”*

---

## 10. Résumé exécutif (pour rapport)

* BayHunter implémente une inversion bayésienne MCMC trans-dimensionnelle classique.
* Son coût est dominé par les appels au forward.
* MIGRATE se situe dans un régime **amorti / data-driven**.
* La comparaison doit se faire **en nombre de simulations**, via PPC et métriques dérivées.
* Le comptage des forwards est **indispensable** pour une comparaison scientifique honnête.


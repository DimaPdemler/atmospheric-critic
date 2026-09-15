# The Discriminator Approach, Explained

*A from-scratch walkthrough of the discriminator direction for measuring the
physical realism of atmospheric forecast fields. Written to be readable if
you've been working on the other (representation-learning) direction and need
to get up to speed.*

---

## 1. The problem we're actually solving

We want to measure the **realism** of forecast weather fields. Not "is this
forecast accurate for next Tuesday," but "does this field *look like* a
physically plausible state that Earth's atmosphere could actually produce."

This is a different question from what standard forecast metrics answer, and the
distinction is the whole point of the project.

### Forecast skill vs. distributional realism

Let $P$ be the distribution of **real** atmospheric states (we use ERA5
reanalysis as a stand-in for reality). Let $Q_{\mu,\tau}$ be the distribution of
fields produced by forecast **model $\mu$** at **lead time $\tau$** (e.g.
GraphCast at 24h).

- **Traditional metrics** (RMSE, CRPS, most of WeatherBench2) compare a forecast
  against the *single* weather state that actually occurred. In distributional
  language, they treat the target as a **Dirac delta** — a single point. They
  reward getting close to that one outcome.

- **Our metrics** compare the *entire distribution* $Q_{\mu,\tau}$ against the
  *entire distribution* $P$. This is a **marginal / distributional** comparison
  over many times, variables, and grid points — not a per-sample, matched-in-time
  comparison.

Here "marginal" just means: pool all the forecast fields into one cloud of
points, pool all the real fields into another cloud, and ask how different the
two clouds are — ignoring which forecast corresponds to which day.

### Why the distinction matters: over-smoothing

The canonical failure mode is **over-smoothing**. ML forecast models minimize
pointwise error (L2), and the L2-optimal prediction under uncertainty is a
*blurry average* of the possible outcomes. So models learn to smear out
fine-scale structure.

- RMSE **rewards** this — a blurry field has lower expected squared error.
- But a blurry field is **physically unrealistic** — real weather has sharp
  fronts, turbulent texture, etc.

We can even measure this artifact directly. The radial power spectrum of
GraphCast/Pangu 24h forecasts matches ERA5 at large scales but loses **50–60% of
wind power** at small scales. This deficit lives in the fine scales, which carry
little total variance, so it is **nearly invisible to RMSE and other bulk
statistics** — yet it is exactly the kind of unrealism we want a metric to catch.

### The goal, stated formally

We want a score
$$
S(Q_{\mu,\tau}; P)
$$
that is **small** when forecasts are statistically indistinguishable from real
states, and **large** when they have detectable distributional artifacts. Its
negative, $-S$, is a "realism score."

> **One philosophical wrinkle.** The Earth only ran once. The "true distribution
> of future weather states" is a distribution of *counterfactuals* ("if Earth ran
> many times"). Real weather is **multimodal** because the dynamics are strongly
> nonlinear. Traditional metrics, which target a single collapsed outcome, are
> effectively unimodal and can't reward a model for correctly capturing multiple
> plausible modes. Our distributional view can.

---

## 2. Two directions (and why we're on the discriminator one)

The paper explores **two** ways to build the score $S$:

| | **(A) Discriminator** | **(B) Representation learning** |
|---|---|---|
| Idea | Train a classifier to tell real from fake | Encode fields, compare latent distributions |
| Tool | ConvNet / linear-probe-on-frozen-encoder | FID (Gaussian-Wasserstein) + MMD |
| Feature choice | **Adaptive** — learned from data | **Fixed** — whatever the encoder gives |
| Status | Works well; the focus now | Largely fails on real forecasts |

The two are a **bias–variance tradeoff**. The fixed-representation approach (B)
commits to a representation up front, which biases it toward certain features and
can leave it **blind** to the artifacts that actually matter. Empirically, the
encoders (MAE, I-JEPA) saturate on synthetic corruptions but **fail to separate
real forecast smoothing** — which is precisely the motivation for leaning on the
adaptive discriminator (A).

The rest of this document is about **(A)**.

---

## 3. The discriminator, from intuition to math

### 3.1 The intuition

If real fields and fake (forecast/corrupted) fields are distributionally
different, then a classifier should be able to tell them apart. If it **can**,
they differ; **how confidently** it can tells you *how much* they differ. This is
the classic idea of a **classifier two-sample test**.

The beauty: we don't hand-pick what "unrealistic" means. The network **discovers**
whatever features separate fake from real — spectrum, texture, cross-variable
physics, whatever is there.

### 3.2 The formal scaffolding: $f$-divergences

An **$f$-divergence** measures how different two distributions are. For a convex
function $f$ with $f(1)=0$, using the convention in the write-up:
$$
D_f(P \parallel Q) = \mathbb{E}_{x\sim P}\left[ f\!\left(\frac{q(x)}{p(x)}\right)\right].
$$

Every $f$-divergence has a **variational (lower-bound) form** — you can estimate
it by *optimizing over a critic function* $T$:
$$
D_f(P \parallel Q) \;\ge\;
\sup_{T\in\mathcal{T}}\left(
  \mathbb{E}_{\hat{x}\sim Q}\big[T(\hat{x})\big]
  - \mathbb{E}_{x\sim P}\big[f^*\big(T(x)\big)\big]
\right),
$$
where $f^*$ is the convex conjugate of $f$. The critic $T$ is exactly our
**discriminator**: it outputs scores that expose the difference between $Q$ and
$P$.

**Our choice of $f$.** We pick $f(u) = u\log u$, which makes the divergence the
**reverse KL**:
$$
D_{\mathrm{KL}}(Q \parallel P) = \mathbb{E}_{\hat{x}\sim Q}\left[\log\frac{q(\hat{x})}{p(\hat{x})}\right].
$$
Its conjugate is $f^*(t) = \exp(t-1)$. Plugging in, the objective the
discriminator $T_\omega$ (a network with weights $\omega$) maximizes is:
$$
\boxed{\;
F(Q,\omega) =
  \mathbb{E}_{\hat{x}\sim Q}\big[T_\omega(\hat{x})\big]
  - \mathbb{E}_{x\sim P}\big[\exp\!\big(T_\omega(x)-1\big)\big]
\;}
$$
At the optimum (over a rich enough network class), $\max_\omega F(Q,\omega)$
**equals** $D_{\mathrm{KL}}(Q\parallel P)$. So training the discriminator both
*produces a scoring function* and *estimates the divergence*.

### 3.3 The practical reality: it's just cross-entropy

Here's the part that trips people up. **We don't actually optimize that exotic
objective in the code.** Instead we train a **plain binary classifier**:

- real samples $x\sim P$ → label $1$
- fake samples $\hat{x}\sim Q_{\mathrm{ref}}$ → label $0$
- ordinary **cross-entropy** loss

Why is that legitimate? Because of a standard result. If the classifier outputs a
logit $v_\omega(x)$ and $\sigma$ is the sigmoid, the cross-entropy optimum is
$$
\sigma(v^*(x)) = \mathbb{P}(Y=1\mid x) = \frac{p(x)}{p(x)+q_{\mathrm{ref}}(x)},
$$
which rearranges to
$$
\boxed{\;v^*(x) = \log\frac{p(x)}{q_{\mathrm{ref}}(x)}\;}
$$
The classifier logit **is the log density ratio**. That is the quantity every
divergence in this family is built from.

**So the two layers connect like this:**

1. **Cross-entropy does the real work** — it gives us a density-ratio estimator
   $v_\omega(x) \approx \log\frac{p(x)}{q(x)}$.
2. **The $f$-divergence formalism is just the recipe** for turning that logit
   into a final scalar score.

Concretely, to get the reverse-KL critic from the trained logit you use one of
these equivalent transforms:
$$
T_\omega(x) = 1 - v_\omega(x),
\qquad\text{or}\qquad
T(x) = -\exp\!\big(-v_\omega(x)\big),\;\; f^*(T(x)) = v_\omega(x)-1.
$$

> **Mental model:** train a normal classifier, read its logit as a log-density
> ratio, then run that ratio through a fixed formula to get the realism score.
> Don't let the two notations ($v$ vs. $T$) fool you — it's one network, two
> readouts.

---

## 4. What do we train against? The reference mixture $Q_{\mathrm{ref}}$

A discriminator trained against **one** forecast model would just learn *that
model's quirks* (its particular smoothing signature), not a general notion of
physics. To avoid that, we train against a **reference mixture** of many fake
sources.

Let $\mathcal{Q}$ be a family of "plausible forecast-like distributions" and
$\pi$ a weighting over them. The reference is the mixture
$$
Q_{\mathrm{ref}} = \int_{\mathcal{Q}} Q\, d\pi(Q),
$$
which just means: to draw a fake sample, first pick a source $Q\sim\pi$, then draw
$\hat{x}\sim Q$. Equivalently,
$$
\mathbb{E}_{\hat{x}\sim Q_{\mathrm{ref}}} = \mathbb{E}_{Q\sim\pi}\,\mathbb{E}_{\hat{x}\sim Q}.
$$

The mixture in practice contains:

- **Real ML forecasts** — multiple weather models,
- **Multiple lead times** — the *same* model at a different $\tau$ counts as a
  *distinct* fake source,
- **Synthetic corruptions** — deterministic degradations of real ERA5 fields
  (see §6).

The training objective and its optimum are just the earlier ones with
$Q\to Q_{\mathrm{ref}}$:
$$
\omega^* = \arg\max_\omega F(Q_{\mathrm{ref}},\omega),
\qquad
\max_\omega F(Q_{\mathrm{ref}},\omega) \approx D_{\mathrm{KL}}(Q_{\mathrm{ref}}\parallel P).
$$

### Honest limitations of the mixture

1. The family $\mathcal{Q}$ is **unknown** — there's no principled way to define
   "all plausible fakes."
2. The weighting $\pi$ is **unknown** — even relative weights between sources are
   hard to justify. (We sidestep this with uniform or hand-chosen weights.)
3. **Averaging hides rare-but-severe failures.** A source that is egregiously
   wrong (say, absurd South-Pole temperatures) but has small weight under $\pi$
   contributes little to the objective, so the discriminator may never learn to
   catch it.

### The adversarial alternative (considered, rejected)

Instead of averaging, target the **worst** source:
$$
\max_{Q\in\mathcal{Q}}\; D_{\mathrm{KL}}(Q\parallel P)
= \max_{Q\in\mathcal{Q}}\;\max_\omega
\left(
  \mathbb{E}_{\hat{x}\sim Q}[T_\omega(\hat{x})]
  - \mathbb{E}_{x\sim P}[\exp(T_\omega(x)-1)]
\right).
$$
This kills the need to weight sources and directly hunts the biggest failure.
But with only a handful of models/corruptions it tends to just **lock onto the
most obviously-broken source and overfit** to it, rather than learning general
physics. So we use the mixture as the pragmatic compromise.

---

## 5. Scoring a new forecast model

After training, we **freeze** the discriminator and use it to score any candidate
forecast distribution $Q_{\mu,\tau}$:
$$
S_{\mathrm{adv}}(Q_{\mu,\tau}) =
  \mathbb{E}_{\hat{x}\sim Q_{\mu,\tau}}\big[T_{\omega^*}(\hat{x})\big]
  - \mathbb{E}_{x\sim P}\big[\exp\!\big(T_{\omega^*}(x)-1\big)\big].
$$

- **Large $S_{\mathrm{adv}}$** → the forecast differs from reality in ways the
  discriminator can detect → *unrealistic*.
- $-S_{\mathrm{adv}}$ is the realism score.

### Two evaluation regimes (important!)

**(i) Train on a model, test on the same model.** Theoretically clean: this is
genuinely the reverse-KL at the high-statistics limit. This is the **defensible**
framing and what the paper mainly targets.

**(ii) Train on many models, test on a held-out new one.** More like asking "are
these samples more like $P$ or like $Q_{\mathrm{ref}}$?" This is riskier:

> **A real failure mode.** If a generator produces **hyper-realistic** samples —
> e.g. it mode-collapses and outputs near-duplicates of a few real fields — the
> score can go **negative** (falsely "more realistic than real"). A
> classifier-based score can't see *lack of diversity*: a delta distribution
> sitting on one real sample looks perfectly realistic to it, even though real
> weather varies. This is a point in favor of the per-model regime (i). (A
> negative score *could* also be read as a useful tell of mode collapse — but that
> needs more theoretical work.)

---

## 6. The corruption library (our synthetic ground truth)

Before trusting the metric on real forecasts (where there's **no** ground-truth
realism ordering), we validate it on **synthetic corruptions**: deterministic
degradations of real ERA5 fields with a tunable **severity** $s$.

The logic: each corrupted sample is *derived from a real state*, so by
construction it's **less realistic** than its source, and $s$ gives a
**ground-truth ordering** of unrealism. A good metric should increase
**monotonically** with $s$. We sweep $s$ over nine levels from $0$ up to
$\mathrm{MAX\_SEVERITY}=2$ (twice the nominal calibration) to probe beyond the
calibrated range.

All corruptions act in **per-channel standardized space** (so noise magnitudes
are comparable across variables with wildly different units — K vs. Pa) and are
**padding-aware** (strip padding → corrupt the real interior → re-pad, wrapping
in longitude, zero-padding in latitude).

**Two families of corruption:**

**Generic degradations** (mimic ML-forecast artifacts, affect all fields):
| Corruption | Effect | Severity → parameter |
|---|---|---|
| Gaussian blur | Over-smoothing | $\sigma \in [0, 1.125]$ |
| High-freq. noise | i.i.d. per-pixel Gaussian | $\mathrm{std}\in[0,0.25]$ |
| GRF noise | Spatially-correlated (FFT) noise | $\mathrm{std}\in[0,0.375]$ |
| Random pixel replace | Pixels → Gaussian samples | $p\in[0,0.3]$ |

**Physically-motivated** (target cross-variable consistency; touch **only** the
wind channels U10/V10, leaving T2M & MSL intact):
| Corruption | Effect | Severity → parameter |
|---|---|---|
| Wind patch shuffle | Permute wind patches | fraction $\in[0,1]$ |
| Wind channel rotation | Rotate wind vectors | angle $\in[0°,90°]$ |

The wind corruptions are the clever ones: they keep **local wind statistics**
and texture intact but **break the physics** — patch shuffle destroys the
alignment between pressure systems and their winds; rotation breaks the
geostrophic relationship between wind direction and pressure gradient. A model
that penalizes these is responding to **genuine physical inconsistency**, not
just visible texture damage.

The discriminator also adds two evaluation-only corruptions — **patch dropout**
(square patches → field mean) and **quantization** (round to a few levels) — plus
a **fieldwise** mode that corrupts a single variable to isolate per-variable
sensitivity.

---

## 7. Why the baselines aren't enough (and what Slack found)

We compare the learned score against hand-designed baselines that act on raw
fields or fixed transforms — all in the same marginal view ($Q_{\mu,\tau}$ vs.
$P$, **not** conditional forecast error):

- **Mean bias** $S_{\mathrm{mean}} = \mathbb{E}_Q[\hat x] - \mathbb{E}_P[x]$ and
  **relative std error** $S_{\mathrm{std}} = \mathrm{Std}_Q/\mathrm{Std}_P - 1$
  — trivial sanity checks; blind to structured artifacts.
- **Unpaired CRPS / energy distance** — a two-sample energy statistic
  $S_{\mathrm{energy}}(Q;P) = \tfrac12 D_E(P,Q)$.
- **Raw MMD** — kernel two-sample test on flattened fields; dominated by global
  Euclidean distance.
- **Raw Gaussian Wasserstein** — FID without an encoder; covariance ill-conditioned
  in high dimensions.
- **Sliced Wasserstein** — random 1-D projections.
- **Zonal energy spectrum** — relative $L^2$ distance between mean zonal spectra;
  discards phase and meridional structure.
- **SCWD** (spherical convolutional Wasserstein) — geometry-aware, but **local**
  by construction.

### The experimental game

The point of the Slack thread (Younes reporting, Joel advising) is to find, **for
each baseline, a corruption it misses but the discriminator catches**. As Joel
put it: we don't need one "metric kryptonite" that defeats *everything* — we need
to show that *each* well-chosen existing metric has *some* blind spot we cover.

What they found:

- **Hemisphere splice** (paste another ERA5 sample's southern hemisphere in):
  **SCWD barely notices** — only the equatorial *seam* lights up, because SCWD's
  probes are **local** and the local distribution only changes at the stitch. This
  is the argument against SCWD's **locality bias** — "it doesn't see the physics of
  the Earth." Global-mean Wasserstein *does* catch it; the **zonal spectrum fails**
  (latitude bands are preserved exactly).
- **Checkerboard pattern** (~4 K, a plausible convnet upsampling artifact):
  invisible to **SCWD and global-mean WD**, but easily caught by **log-spectral L2**.
- The **raw discriminator** responds well to all of these. Its one clear weakness
  is **spectrum-matched noise** (a response exists, but it's weak).

### Other things in the thread

- **New train/test split** (from the normalizing-flow paper): train on days 1–15,
  test on days 20–26, with **5-day buffers** to prevent temporal leakage. For
  forecast-vs-ERA5, samples are **temporally matched** (keep only forecasts with a
  matching ERA5 timestamp).
- **Attribution** (Integrated Gradients / "Axiomatic Attribution for Deep
  Networks"): which grid cells drive the decision. Usually the **oceans**,
  sometimes high-frequency land (the **Rockies**). For pixel-replace it correctly
  highlights the flipped pixels. For hemisphere splice, masking the seam shifts
  attention to **polar regions** — hinting it learned some **seasonal** signal.
  Pitched as a **shortcut-detection** tool.
- **Visualizations** for the manuscript (Joel signed off on these):
  - **Logit histograms** per lead-time / corruption-strength — a clean visual of
    how far the discriminator pushes the two distributions apart. (Joel: make one
    per experiment, then select.)
  - **Corruption-vs-severity** curves (pixel-replace dominates, all trend up) and
    **lead-time** curves (small data makes the ERA5-vs-ERA5 "null result" drift
    negative; curves track the spectral-error shape).
  - **Latent PCA / t-SNE colored by lead time or class** — to argue the classes
    are *not* cleanly separable in the encoder's latent space (why FID/MMD
    struggle) but *become* separable after the discriminator's nonlinear head.
- **SFNO + linear probe** — planned, to "regularize the divergence"; the bridge
  between the two directions (a physics-aware encoder as the discriminator
  backbone).

---

## 8. One-paragraph summary

Train an **adaptive classifier** to separate real ERA5 fields (label 1) from a
**mixture of fakes** — forecasts at various lead times plus synthetic
corruptions (label 0). Cross-entropy training makes the classifier logit an
estimate of the **log density ratio** $\log\frac{p}{q}$; a fixed $f$-divergence
formula turns that logit into a **reverse-KL-based unrealism score**
$S_{\mathrm{adv}}(Q_{\mu,\tau})$. Because the classifier *learns* what makes a
field unrealistic, it catches artifacts (over-smoothing, broken wind–pressure
physics, hemisphere splices, checkerboards) that each hand-designed baseline —
SCWD, zonal spectrum, Wasserstein — individually **misses**. The experimental
program is (1) validate monotonic response on ground-truth synthetic corruptions,
and (2) show, baseline by baseline, a failure case the learned metric covers.

# Unified multimodal world model with task-aware context compression

A single Qwen3.5-2B predictor learns the future distributions of entity states, relational predicates and event chains. It is conditioned on the previous five seconds of history:
- entity-centred structured text, through the native tokenizer and word embeddings;
- sensor observations, as spatially and temporally bound embeddings (RGB through the native visual path).

A smaller Qwen selector learns which historical evidence to keep under a context budget. It is supervised by how candidate evidence sets change the predictor's loss on the actual future. The two models are trained cooperatively, by alternation.

Unification means three things are shared by every world variable: the backbone, the conditioning context, and the query-to-probability interface. Code handles candidate construction, serialization, budget accounting and answer decoding.

**Research claim.** A unified predictor over entity text and images can learn, from future-prediction loss, a selection of history that:
- keeps probabilistic prediction quality and calibration for states, predicates and events within a limited context;
- does better than simple selection at the same budget.

Alternating small/large updates may add further gains.

---

## 1 Objective

The paper's contribution is two connected mechanisms:

1. **One probabilistic model of one future.** Numeric states, relational predicates, events and their times are all targets of the same future world. Entities, roles and time stay linked, and every target type is scored by the same language-model head over valid answer codes.
2. **Evidence value from prediction loss.** The predictor's loss on the realized future measures how valuable each piece of historical evidence is. A lightweight model learns to estimate that value from the history alone. As the predictor adapts to compressed inputs, the estimates are refreshed.

Evidence value depends on context. Records can be redundant with each other, or useful only together. Supervision therefore keeps both:
- set-level losses of complete candidate sets;
- local deletion effects of individual records or record groups.

A domain foundation model is the long-term direction. This paper's claim is narrower: task-aware compression for multimodal world prediction, and its training method.

---

## 2 Data

All sources are joined by entity and tick. One tick is 0.1 s, the history is 5 s (50 ticks), and the horizons are 0.5, 1, 2, 3 and 5 s (ticks 5, 10, 20, 30, 50).

| Source | Content | Role |
|---|---|---|
| `canonical.jsonl` (truth frames, roster, world-truth graph base and deltas, objective events) | entity motion, roster, predicate assertions and changes, events | numeric history and labels; relation history |
| `communication_state.jsonl` | flows, handover (serving station, switching state), link quality | communication history and targets |
| `compute_state.jsonl` | resource requests, node capacity, task and queue states | compute history and targets |
| `predicate_truth.jsonl`, `events.jsonl`, `event_occurrences.jsonl` | producer predicates and events with bindings and times | future predicate and event labels, read with their original semantics |
| `event_trace.jsonl`, `event_realization.jsonl` | scheduled activations and execution records | process history; future actions not yet known at the origin are targets |
| capture embeddings (RGB, depth, segmentation, LiDAR) | per observer and tick, with calibration and entity bindings | sensor evidence |

**Scope of targets.** The five pilot predicates are examples only. The target schema covers every producer predicate and event with valid labels, including:
- compute: queued, offloaded, migrating, execution_failed, over_allocated, node_unavailable;
- communication: retransmission, handover, latency and bandwidth thresholds, session interruption;
- airspace and safety relations;
- scheduled lifecycle events and their consequences.

A predicate keeps its original bindings. Communication predicates, for example, bind flow or handover objects rather than UAVs.

**Label validity.** A label enters the loss when it is defined, correct, and compatible with the state it describes. Labels are masked when they are:
- undefined (no producer value, or an unknown value);
- known to be incorrect;
- incompatible, for example produced by a producer version that does not match the state it describes, as with the earlier `link_degraded`/`link_unavailable` mismatch.

Masking applies to individual labels, or to a whole family only when the whole family fails one of these conditions.

A family whose value never changes in the corpus remains a valid target. Consistently true or false labels supervise the prediction of stable states. Class imbalance is handled in the objective (§6: group weights and reweighted resampling) and in reporting (per family, separating stable and changing cases), not by excluding targets. Unknown labels keep their validity masks and never become false.

**Episodes.** The pool is 210 canonical parents: 173 original captures and 37 v14 recaptures, mapped in `configs/capture_sources.json`. The final models train on all 210. Arms are paired with their factual parents and used only for intervention and counterfactual evaluation (§8).

---

## 3 Records, candidate units and serialization

### 3.1 Record

```text
record = {id, time_s, bindings, field, value, status}
status ∈ {observed, unknown}
```

- `time_s` is relative to the prediction origin.
- `bindings` keep role names, e.g. `{actor: u1, corridor: c1}`.
- `field` is the original field or predicate name.
- `value` is a number, vector, category, boolean or entity reference.
- Units and coordinate frame are declared once in the window header.

Three states must stay distinguishable in the serialized text: *unknown*, *observed false*, and *removed by selection*. Removed records are simply absent. Whether the text also marks that something was omitted (for example "3 records omitted") is a design variable of the method.

### 3.2 Candidate construction and reconstructible compaction

At 10 Hz, a full 5 s history is far larger than any useful budget. Two measured windows:
- 22 moving entities: about 40k tokens of motion alone;
- 54 moving entities: about 100k tokens of motion alone.

Predicate-evidence records add 2.8k–10.5k entries per window.

Before learned selection, only **redundancy that can be undone** is removed. Nothing that could be useful evidence is discarded before the selector can assess it. The property is stated and checked as follows.

**Quantization (lossy, declared).** Each numeric field f has a declared quantum ε_f, for example 0.01 m, 0.01 m/s, 0.1 ms, or a resource-specific unit. The record stream is first quantized:

```text
x̃_t = ε_f · round(x_t / ε_f)
```

This step is lossy, by at most ε_f/2 per value. ε_f is chosen finer than the resolution of the corresponding answer intervals (§4.3), and both are stated in the window header.

**Compaction (lossless relative to the quantized stream).** For every addressed stream (entity field, relation evidence value, communication or compute quantity, observation attribute), the compacted form consists of:
- the stream's presence intervals, with explicit start and end times, so that appearance and disappearance are never implied;
- the initial quantized value at the first present tick;
- the time-stamped change points;
- a reconstruction rule R, named in the header, that regenerates every quantized sample from these.

The compacted form is admissible only if R reproduces the quantized stream exactly:

```text
R(compact(x̃))_t = x̃_t   for every present tick t in the window.
```

This equality is verified for every window. A stream that fails it keeps all of its samples as candidates.

**Reconstruction rules:**
- **Categorical, boolean, entity-valued streams, and numeric streams with exact holds:** hold the last value. Change points are ticks where x̃_t ≠ x̃_{t−1}.
- **Motion:** a piecewise rule over knots (t_k, p̃_k, ṽ_k). Each knot stores position and velocity. Between knots, velocity is held and position follows

  ```text
  p̂_t = p̃_k + ṽ_k (t − t_k)
  ```

  Both are then quantized again. A new knot is placed at the first tick where p̂_t or ṽ_k differs from p̃_t or ṽ_t. The first present tick is always a knot. Knots carry their tick, so timing is recovered exactly.

**Evidence below unchanged predicates.** A predicate's boolean value can stay constant while its evidence changes: distances to a threshold, resource levels, queue lengths, link quality, observations. Those evidence quantities are separate addressed numeric streams with their own change points. They are never dropped because the predicate's truth value is unchanged.

Only evidence refreshes that reproduce identical quantized values at the same address are reconstructible and removed.

**Fixed block.** The fixed block contains:
- each present entity's identity and type;
- the origin-time state of the entities the query set addresses: queried entities, and the entities a queried relation binds, directly or as the owner of a bound object such as a flow, task or queue (frame line, plus the link or compute line for communication or compute predicates);
- the origin-time state of the queried relations.

The origin-time state of other entities and relations is a selection candidate like every history record. In an origin line, `flags=[...]` lists the recorded true booleans and `off=[...]` the recorded false ones. A boolean in neither list was not recorded and is unknown, never an implicit false.

Every compacted change point (motion knots, value changes, evidence-value changes, events) is a selection candidate. The candidate count after compaction is measured per window. If it still exceeds what the selector can read at once, the selector reads it in chunks; candidates are never pre-filtered.

### 3.3 Selection units

The selection unit is a **record group**: a set of records that only make sense together. Examples:
- one motion record (position and velocity at one instant);
- one relation change with its bound objects;
- one communication or compute sample.

Entity identity and type, and the origin-time state the query set addresses (above), form the fixed block and are always kept. Captures are selection units too (§3.5). Later, statistical summaries (for example "duration of the current connection state", "velocity change over the last 2 s") can be added as extra candidates. The selector would then choose among the raw records, a summary, or nothing.

### 3.4 Serialization

Kept groups are merged deterministically by entity, then time, then field. Repeated headings are removed and no free-form rewriting happens.

Example of the kept text. Coordinates are relative to a declared per-window origin point o, which here coincides with u1's position at t=0:

```text
history_s=5 horizons_s=[0.5,1,2,3,5] frame=enu_m(origin=o) quanta: pos 0.01 m, vel 0.01 m/s, latency 0.1 ms
reconstruction: motion=velocity-hold knots; other fields=hold-last-value
entity u1 type=aircraft
  t=0     position=[0.00,0.00,28.00] velocity=[-0.63,4.71,0.00] latency_ms=44.5 handover_active=false
  t=-0.5  position=[0.31,-2.35,28.00] velocity=[-0.63,4.71,0.00]
objects f1=data_flow(u1) h1=handover(u1) c1=airspace_corridor
relations
  t=0 airspace.aircraft_occupies_airspace_corridor actor=u1 corridor=c1 value=false
```

Aliases (u1, f1, c1, …) are assigned per window by a deterministic rule. Queries and answers use the same aliases.

### 3.5 Sensor evidence

- **RGB:** enters through Qwen3.5's native visual path. Cached post-merger visual tokens are 98 per frame at the current preprocessing, each with its `image_grid_thw` for the native multimodal position encoding.
- **Depth, segmentation and LiDAR:** reuse the existing encoders. Each needs a learned projection into the 2048-dimensional hidden space, with position indices tied to the corresponding RGB frame.
- **RGB-only cameras:** the fixed overview camera of every parent gives RGB every 0.5 s with no depth, segmentation or LiDAR. Its frames enter with RGB tokens only and their own modality embedding. Absent modalities occupy no positions.
- **Bindings:** every observation carries its time, observer, modalities, calibration and entity associations. These appear as a short text header immediately before the visual positions.

**Captures are selection units.** Every actual capture of every observer in the 5 s window is a candidate: four-modal UAV captures and RGB-only fixed-camera captures. No limit is set on observers, frames or modalities; the selector chooses text records and captures jointly under one total budget. For each capture it reads:
- a metadata line (observer, modalities, time, pose, field of view, the entities in view and the relations binding them);
- one 2,631-dimensional summary of the frozen features, projected into one embedding. Absent modalities are marked, not zero-filled.

A text-only stage, in which a fixed capture packet is shared by all candidate text sets, remains available as an option.

**Constraint.** The cached visual tokens fix the vision preprocessing and assume a frozen vision tower. Changing either later needs the raw captures.

---

## 4 Probability interface

### 4.1 Queries

```text
query  = {bindings, target, horizon or interval, domain}
answer = {query_id, support, probabilities}
```

The query set Q of a window covers states, predicates, events and times for the relevant entities. Context is selected once per window, or per task group, and shared by all queries.

### 4.2 Answer codes and scoring

Every domain D_q is a finite set of answer codes. The codes form a prefix tree T_q: each code is a root-to-leaf path, and no valid code is a proper prefix of another.

**Chosen construction: locally normalized prefix tree.** At an internal node with prefix a, the native LM-head logits ℓ_θ(· | C, q, a) are restricted to the node's valid children ch(a) and normalized there. A code's probability is the product along its path:

```text
π_θ(t | a) = exp ℓ_θ(t | C,q,a) / Σ_{t' ∈ ch(a)} exp ℓ_θ(t' | C,q,a),   t ∈ ch(a)
p_θ(c | C,q) = Π_{j=1..L(c)} π_θ(c_j | c_<j)
```

Because every node's restricted softmax sums to one, Σ_{c∈D_q} p_θ(c | C, q) = 1 by construction. This is the probability used everywhere:
- as the training likelihood;
- in all reported scores;
- as the conditional factor in joint mode (§4.4).

This construction is generally **not** equal to normalizing unrestricted sequence scores over complete valid codes. That alternative is not used. When the domain is a single level (booleans, small categories, the six flip-time classes), the two coincide.

**Cost.** A full distribution needs one forward step per internal node of T_q. Prefixes are shared through the KV and recurrent caches.

| Target | Domain |
|---|---|
| Boolean predicate | {false, true} |
| Category or candidate entity | the valid categories or entities |
| First-flip time | (0,0.5], (0.5,1], (1,2], (2,3], (3,5], >5 s |
| Event | occurrence × type × participants × time interval, including "none in interval" |
| Numeric state | interval codes (4.3) |

### 4.3 Numeric states as interval codes

A numeric target is coded as a residual r = y − y_ref from a deterministic reference computed by code. Inputs keep the quantized raw values with units (§3.2); intervals apply only to outputs. Chronos is the precedent for training quantized numeric tokens with LM cross-entropy.

**Reference: R0 (decided).** The reference is the currently observed value, y_ref = x₀; for example, position at the origin.

R1 (constant-velocity extrapolation for position, y_ref = p₀ + v₀·h) was considered and not adopted. It would change the parameterization of the prediction, not the information available: v₀ is already in the input. For entities that keep their velocity, it concentrates probability near zero. Under either reference, the code space must represent substantial departures, such as turns, stops or reroutes. The support below is sized for the largest physically possible residual under the chosen reference.

**Observation model.** The simulator stores y at a finite storage precision δ_f. The predicted object is a discrete distribution over a partition {I_k} of the residual axis. The observed answer is the index k* with r ∈ I_{k*}. The training likelihood and the primary score are the discrete negative log-likelihood −log p_θ(k* | C, q).

- **Near-zero interval.** I₀ = (−ε_f/2, ε_f/2] means "within ε_f/2 of the reference". It is an interval, not a point mass.
- **Exact equality** at storage precision is a different event. It is included as its own code only when its probability is itself of interest, for example the compact model's "no deviation" probability. It is then the interval of width δ_f around the reference, with that declared meaning. No Dirac atom is used anywhere.
- **Other intervals.** Interval edges grow log-spaced in |r| from ε_f/2 up to the support bound R_{f,h}, separately for each sign.
- **Code tree.** Codes are hierarchical (coarse interval, then sub-interval), using existing single tokens, under the locally normalized tree of §4.2. Vector fields are coded component by component in one tree, so their dependence is represented.

**Support and tails.**
- **Bounded support (preferred).** Where physics bounds the residual, the partition covers exactly [−R_{f,h}, R_{f,h}], and every interval is bounded. Examples:
  - per-category maximum speed and acceleration times horizon, adjusted for the reference;
  - the link-model latency range.

  A label outside the declared support is a data error and is reported by the audit, never silently clipped.
- **Explicit tails (otherwise).** Two overflow codes, (−∞, −R) and (R, ∞), each carry a declared tail density. For example, exponential:

  ```text
  f(r) = π_over · λ · e^{−λ(|r| − R)},   |r| > R
  ```

  with λ fitted on development data. The discrete likelihood needs only π_over. The tail density matters only for continuous scores.

**Continuous scores.** Within each bounded interval the density is uniform:

```text
f_θ(r) = Σ_{k bounded} p_θ(k) · 1[r ∈ I_k] / |I_k|  +  tail terms
```

With bounded support, or with exponential tails, the CDF is piecewise linear or exponential. CRPS then has a closed form, and its value depends on the declared within-interval and tail densities. Comparisons with continuous predictors, such as the compact model, are made on the common observation: their CDFs are integrated over the same intervals, which gives discrete NLL and ranked probability score on an identical outcome space. CRPS is reported alongside, with the densities stated.

### 4.4 Marginal and joint modes

The same parameters serve two attention patterns over the query block that follows the shared context.

- **Marginal mode.** Each query attends to the context and its own code prefix only. All queries are answered in parallel and give p_θ(y_q | C).
- **Joint mode.** Queries are ordered by time first (all targets at 0.5 s, then 1 s, …; within a time step by a fixed entity/field order). Each answer is conditioned on all earlier answers:

```text
P_θ(Y | C) = Π_j P_θ(Y_j | C, Y_<j)
```

Training uses teacher forcing on the realized future. Sampling K joint futures uses ancestral sampling over the shared context prefix; K is a runtime choice. First-flip and event-time targets are their own queries in this order, because a predicate can flip and flip back within an interval. Consistency between a sampled flip time and the sampled per-horizon states is checked by the rules where it is defined.

### 4.5 Latent z (optional extension)

```text
v = E_vision(I),   h = F_θ(H_text, v),   p(Y | H, I) = ∫ p(z | h) p(Y | h, z) dz
```

- v is the observed image embedding.
- h is the backbone's context representation.
- z is an optional latent variable shared across one sampled future. It needs no labels and its dimensions need no names.

Native autoregression already represents coherent joint futures. z is studied only if joint-mode sampling proves too costly, or too incoherent under joint scores. In that case z would condition a lightweight output decoder so that the backbone runs once per window.

---

## 5 Supervision from the realized future

**Numeric labels** are the actual future states, coded as in 4.3.

**Predicate and event labels** follow their production sources. Producer records (`predicate_truth`, `events`, `event_occurrences`, process records) are read directly with their bindings, times and order. Rule-defined predicates are derived from the complete future states they require.

Three questions are kept separate:
- how a label enters the pipeline;
- whether the complete simulator state determines it;
- whether the selected history determines it.

Even a fully determined predicate is a probabilistic target given limited history.

**Censoring.** When the future record ends before 5 s, the first-flip loss uses the probability mass of every category still compatible with the observation: −log Σ_{v compatible} p_θ(v).

**Causality.**
- Inputs contain only records known at the origin, plus action conditions already fixed at that time.
- Scheduled future actions that are not yet known are targets.
- The selector never sees the future.

---

## 6 Objective

**Notation.**
- H: compacted candidate history (record groups).
- S ⊆ H: the kept groups.
- I: the window's selected captures. With joint selection, captures are part of S; I denotes them separately only in a text-only stage, where a fixed packet is shared by all candidate text sets.
- C(H, S, I): the assembled input.
- Q: the query set, partitioned into target groups g (numeric state, predicate, first-flip time, event, …).

**Prediction loss.** The loss is a weighted average over groups of the mean per-query loss within each group:

```text
L_θ(S; I) = Σ_g w_g · (1/|Q_g|) Σ_{q∈Q_g} ℓ_q(θ; C(H,S,I))
ℓ_q = − log p_θ(y_q | C, q)                 (fully observed labels; numeric y_q = observed interval k*)
ℓ_q = − log Σ_{v∈V_q} p_θ(v | C, q)         (censored labels; V_q = compatible answers)
```

Averaging within each group before weighting stops numerous unchanged fields from dominating. Rare transitions may be oversampled during training. The probabilistic loss is reweighted by the inverse sampling ratio, so predicted probabilities stay calibrated to the natural distribution.

**Budget.** The hard constraint counts actual input positions:

```text
N_fixed + N_text(S) + N_frame(S) + N_visual(S) + N_query(Q) + padding ≤ B = 20,000
```

- N_fixed: background (conventions, queried predicate definitions, answer codes), identity lines and the fixed origin state.
- N_frame, N_visual: frame headers and visual positions after processor expansion, for the modalities each selected capture actually has.
- N_query: the label-free query reserve (the longest code of every query's answer domain), so the realized answers never change the evidence capacity.

Reports give both the text compression ratio and the total input cost.

**Selection target.**

```text
S*(H, Q, B) = argmin_{S: budget} L_θ(S; I)
```

**Evidence utility** of group i given kept set S:

```text
u_i(S) = L_θ(S \ {i}; I) − L_θ(S; I)
```

Loss differences are computed with the predictor, queries, realized future and full candidate universe held fixed. The selected text and captures vary; in a text-only stage the capture packet is fixed too. Utilities are context-dependent (redundancy, synergy) and are always recorded together with the set they were measured in.

**The realized-future caveat.** Utilities are computed against the *realized* future. What the selector should learn is expected value given the history:

```text
U_i(H, Q, B) = E_{Y | H}[ u_i ]
```

The minimizing set for one realized future overfits that particular outcome (winner's curse). Selector targets are therefore soft:
- a Boltzmann distribution over evaluated candidate sets, P(S) ∝ exp(−L_θ(S)/τ);
- the per-group inclusion probability it induces.

Training on many windows then averages over outcomes.

---

## 7 Worked example

Three optional groups, each 20 tokens, with 40 tokens of optional budget:
- A: the latest motion record;
- B: the communication history;
- C: an older, repeated position.

All values are constructed for illustration only.

| Kept set | Tokens | p(true position interval) | p(true predicate value) | Mean NLL |
|---|---:|---:|---:|---:|
| ABC | 60 | 0.80 | 0.75 | 0.2554 |
| AB | 40 | 0.79 | 0.75 | 0.2617 |
| AC | 40 | 0.80 | 0.45 | 0.5108 |
| BC | 40 | 0.50 | 0.75 | 0.4904 |

The deletion effects from ABC are:
- u_A = 0.2350;
- u_B = 0.2554;
- u_C = 0.0063.

With τ = 0.05, the Boltzmann weights over the within-budget sets are 0.983 for AB, 0.010 for BC and 0.007 for AC. The induced inclusion probabilities are 0.990 for A, 0.993 for B and 0.017 for C.

The label says "keep A and B for this history, query set and budget". It does not say that old position records are worthless in general.

---

## 8 Datasets

| Set | Content | Use |
|---|---|---|
| World prediction | compacted history, image packet, query set, realized labels; full and randomly reduced histories | train the predictor |
| Compression evaluation | window, budget, candidate sets, per-group losses, position counts | selection supervision |
| Selector training | history, query set, budget, soft set and inclusion targets | train the selector |

**Selector example:**

```json
{"window": "L1-2_v1__seed01@75", "visual_packet": "window_visuals", "budget_positions": 40,
 "candidates": ["A", "B", "C"], "queries": ["u1.position@1s", "f1.latency_above_threshold@1s"],
 "evaluated_sets": [{"keep": ["A","B"], "loss": 0.2617, "positions": 40},
                    {"keep": ["A","C"], "loss": 0.5108, "positions": 40},
                    {"keep": ["B","C"], "loss": 0.4904, "positions": 40}],
 "inclusion_target": {"A": 0.990, "B": 0.993, "C": 0.017}}
```

Three different quantities must not be confused:
- `u_i`: an offline loss change;
- `p_keep`: the selector's retention probability;
- `probabilities`: the predictor's distribution over future answers.

**Windows.** Every parent gives windows at a fixed origin stride. Sampling covers:
- all entity types and horizons;
- stable and changing states, relation formation and dissolution;
- windows around producer events.

**Training scope.**
- **Development and model selection:** temporary holdouts within the parent pool, or cross-validation grouped by scenario family (L1–L6, X1–X6). These results guide settings only.
- **Final models:** once settings are fixed, the predictor and the selector are trained on all 210 canonical parents.

**ARM evaluation.** ARM provides paired intervention and counterfactual evaluation of the final models. Each arm is scored against its factual parent. Arms never fit predictor, selector or calibration parameters. Results are aggregated per parent, and probabilities refer to the intervention distribution, not to natural frequencies.

ARM measures branch-future prediction against parent backgrounds the models have seen. It does not establish generalization to unseen scenario families, and no such claim is made from it.

---

## 9 Cooperative alternating training

Each step names the model being trained and what is held fixed.

1. **Predictor.** Fine-tune the 2B model (LoRA) on entity text and sensor embeddings from the start, over all target groups.
   - Training mixes full compacted histories and randomly reduced histories at several budgets. This keeps later loss differences in-distribution: a predictor that never saw reduced inputs would make every deletion look harmful.
   - Marginal and joint modes are trained together.
2. **Selection supervision (predictor frozen).** For each sampled window, evaluate candidate sets of text records and captures with the realized future, and store their losses, per-group losses and position counts. Every scored set fits the total budget. Sets are generated by:
   - the simple policies, and random mixed subsets of a pool (the policies' selections plus a random reduction at twice the budget);
   - block replacement inside the budget (a block of the best set is replaced by a random refill from the rest of the pool);
   - complementary text–visual groups: a kept capture, the kept records of the entities it shows, or both;
   - swaps and removal effects around the best set.

   Every parent gets a window before any parent gets a second.
3. **Selector (predictor frozen).** Qwen3.5-0.8B reads the candidates in chunks of 8,192 tokens. Each chunk starts with the same prefix: budget, entity roster, the full definitions of the queried relations, the query set and the fixed lines. A bidirectional head over all unit representations of the window (plus the budget) exchanges information across chunks and gives p_keep.
   - **Loss:** binary cross-entropy against the soft inclusion targets (Boltzmann weights over the evaluated within-budget sets), plus a pairwise ranking loss over the measured removal effects.
   - **At inference:** units are packed into the budget in decreasing p_keep, with headings on demand.
4. **Predictor adaptation (selector frozen).** Continue training on the selector's choices, mixed with random text/capture reductions so the predictor doesn't come to rely on one selection policy.
5. **Alternation.** Steps 2–4 repeat as short rounds (a quick closed loop, `unified/scripts/run_loop.sh`). Each round's selector is warm-started from the previous one and trained on the supervision of all rounds so far. Each round's predictor continues from the previous one. A small fixed DEV evaluation after every round compares the selector with the simple policies.

**Selector model.** Qwen3.5-0.8B-Base, the smallest Qwen3.5 sibling with the same tokenizer, so token counts agree.

---

## 10 Questions the evaluation must answer

- **Selection.** At equal budget, with the same predictor and image packet, does learned selection beat simple selection (recency, uniform time, change-point), and how close does it come to full context?
- **Selector capacity.** Does a small structured scorer trained on the same targets match the language-model selector?
- **Alternation.** Does refreshing selection supervision after predictor adaptation improve on a fixed selector at comparable training effort?
- **Images.** What does sensor evidence add, compared with the same text records alone?
- **Previous pipeline.** For the targets they cover, how does the unified model compare with the compact state model, the GRU, and state sampling followed by rules?

**Measures:**
- loss–budget curves per target group (NLL; CRPS and errors for numeric states; Brier score and reliability for predicates and events);
- joint-sample scores (energy and variogram) and rule-consistency rates for joint futures;
- cost in input positions and end-to-end latency (selector, visual encoding, predictor).

---

## 11 References

**Project sources:**
- `Dataset/world_model/model/causal_input.py`, `query_head.py`: field reader and query addressing.
- `Dataset/world_model/predicate_stage/supervision.py`: horizons, first-flip and censoring labels.
- `Dataset/world_model/model/multimodal.py`, `train_state.py`: sensor encoders and the feature cache.
- `Dataset/world_model/configs/capture_sources.json`: parent and arm capture mapping.
- `aw_data/compute_comm_supplement/<episode>/{compute_state,communication_state,predicate_truth,events}.jsonl`
- `aw_data/render_ready_episodes_capture_filtered/<episode>/event_trace.jsonl`

**Methods:**
- LLMLingua / LongLLMLingua / LLMLingua-2: a small model compresses context for a large one; offline selection labels.
- INVASE; L2X: selection learned from downstream loss.
- Chronos; LLMTime: numeric tokenization for probabilistic forecasting.
- Sohn et al. 2015 (CVAE): conditional latent variables for structured outputs.
- Bommasani et al. 2021: the foundation-model scope.
- Qwen3.5 model card and Transformers interface.
- Kev 0.5B: a Qwen-0.5B constrained-decision prototype in the style of JEV (link supplied by the user, not independently verified).

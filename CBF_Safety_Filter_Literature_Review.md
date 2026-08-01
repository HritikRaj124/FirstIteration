# Literature Review Summary: Control Barrier Function Safety Filters
### Prepared for Thesis ST02 — Runtime Safety Filters for Humanoid Robot Manipulation (Unitree G1)

This document summarizes the methodology, key equations, and thesis-relevance of each paper you've collected. Papers are ordered from foundational theory → CBF-QP mechanics → manipulator-specific → humanoid/imitation-specific → review/survey material.

---

## 1. Ames, Xu, Grizzle, Tabuada (2017) — *"Control Barrier Function Based Quadratic Programs for Safety-Critical Systems"* (IEEE TAC)

**Role in your thesis:** The foundational reference — defines the CBF-QP formulation your Sprint-1 roadmap is built around.

### Methodology
The paper's core contribution is unifying **safety** (invariance of a set) with **performance** (stabilization) through a single real-time QP. It proceeds in three stages:

1. **Barrier functions for autonomous (uncontrolled) systems.** Given ẋ = f(x) and a set C = {x : h(x) ≥ 0}, two barrier function types are defined:
   - **Reciprocal Barrier Function (RBF):** B(x) → ∞ as x → ∂C. Classical condition: Ḃ ≤ α(1/h(x)) for a class-K function α.
   - **Zeroing Barrier Function (ZBF):** h(x) → 0 as x → ∂C. Condition: ḣ(x) ≥ −α(h(x)) for an *extended* class-K function α — this is the branch your thesis notes cite (Zeroing Barrier Function branch).
   - Both guarantee forward invariance of C via Nagumo's theorem / the Comparison Lemma.

2. **Control barrier functions (CBFs).** Extending to control-affine systems ẋ = f(x) + g(x)u:
   - **Zeroing CBF (ZCBF)** condition:
     ```
     sup_{u∈U} [ Lf h(x) + Lg h(x) u ] ≥ −α(h(x))
     ```
     Any Lipschitz controller u(x) satisfying this pointwise renders C forward invariant (Corollary 2).
   - For **relative degree r ≥ 2** (i.e., Lg h(x) = 0 up to order r−1, meaning control doesn't appear until the r-th derivative), a higher-order construction Br(x) is built by composing h with a saturation-like function H(·), reducing the problem back to relative degree 1. This is the theoretical seed of the High-Order CBF (HOCBF) constraints you'll need for humanoid joint/torque-level safety (since most useful G1 constraints — collision avoidance in Cartesian space, joint limits under torque control — are relative degree 2).

3. **The CLF-CBF-QP.** Combining an Exponentially Stabilizing CLF V(x) (performance) with a ZCBF/RCBF h(x) (safety) into:
   ```
   min_{u, δ}   uᵀH(x)u + p·δ²
   s.t.  Lf V(x) + Lg V(x)u + c₃V(x) ≤ δ      (CLF constraint, relaxed by slack δ)
         Lf h(x) + Lg h(x)u + α(h(x)) ≥ 0      (CBF constraint, hard)
         A₀u ≤ b₀                               (input constraints, e.g. actuator limits)
   ```
   The **critical design choice**: the CLF (performance) constraint is relaxed with a slack variable δ, while the **CBF (safety) constraint is never relaxed** — this hard/soft split is the template for how your thesis should treat "task tracking" vs. "safety" in the G1 filter. (Note: OSCBF later revisits this by instead slack-relaxing the CBF constraint itself when infeasibility must be handled — see Paper 5.)

4. A closed-form solution for u*(x) is derived via KKT conditions when the CBF is relative-degree 1 and unconstrained in u, giving a Lipschitz-continuous feedback law (important for well-posedness of the closed-loop ODE).

### Key equations to cite
- ZCBF condition (their Eq. 18/Def. 5): ḣ(x) ≥ −α(h(x))
- CLF-CBF-QP (their "CLF-CBF QP" eq., Section IV-B)
- Higher relative-degree CBF construction (Prop. 4)

### Relevance / gap for your thesis
This paper assumes single-input-affine dynamics with scalar h(x); it doesn't address multi-constraint QPs, task-space/joint-space consistency, or humanoid whole-body dynamics — that's exactly the gap OSCBF (Paper 5) and your thesis fill.

---

## 2. Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada (2019) — *"Control Barrier Functions: Theory and Applications"* (ECC)

**Role:** Tutorial/survey consolidating CBF theory post-2017, useful for your thesis's theory chapter and for citing the "why ZBF over RBF" argument cleanly.

### Methodology
- Restates Nagumo's theorem as the historical root of set invariance conditions (ḣ(x) ≥ 0 on ∂C ⟹ invariance).
- Traces the lineage: Nagumo (1942) → barrier certificates (2000s, Prajna et al.) → CBFs (Ames et al. 2014, 2017).
- Surveys applications: adaptive cruise control, bipedal locomotion, multi-robot systems, quadrotors — gives you a "related work" bank of prior CBF deployments across robot classes for your introduction/related-work chapter.
- Reiterates the ZCBF definition and CBF-CLF-QP as the standard tool, and discusses extensions (higher relative degree, robustness to disturbances).

### Key equations
Same Nagumo/ZCBF conditions as Paper 1, presented more pedagogically — good as the "primer" citation before diving into Ames et al. 2017 for rigor.

### Relevance for your thesis
Use this as the citation for general CBF background/motivation paragraphs; use the 2017 TAC paper for the actual equations and proofs you build on.

---

## 3. Rauscher, Kimmel, Hirche (2016) — *"Constrained Robot Control Using Control Barrier Functions"* (IROS)

**Role:** Early manipulator-specific CBF-QP application; a useful "simple case" reference before OSCBF's complexity.

### Methodology
- Considers a general control-affine robotic system ẋ = f(x) + G(x)u, with a **nominal controller** u_nom(x) (independent, arbitrary — e.g. an existing impedance or tracking controller) designed for performance only.
- The CBF-QP is structured as an **add-on / retrofit layer**:
  ```
  min_u  ‖u − u_nom(x)‖²
  s.t.   Lf h_i(x) + Lg h_i(x) u ≥ −α h_i(x),   for each constraint i = 1...N
  ```
  This is the **minimally-invasive safety filter pattern**: rather than co-designing safety and task control, the QP filters an already-designed nominal command. This is directly analogous to what your thesis calls a "runtime safety filter" — the nominal policy (e.g. teleoperation command, RL policy, or scripted motion) is unmodified except at the safety boundary.
- Demonstrates **multiple simultaneous constraints** (workspace boundaries, moving obstacles) stacked as rows in one QP — validates that CBF-QPs scale to N constraints without redesigning the controller, which is relevant since your G1 model will need dozens of constraints (self-collision pairs, joint limits, etc.).
- Experimentally validated on a redundant anthropomorphic manipulator with both static and moving Cartesian constraints, combined with an impedance controller (i.e., compliant robot behavior + hard safety guarantee).

### Key equations
- Multi-constraint QP retrofit structure (their Eq. in Section IV) — this is essentially the "speed-limiting filter" structure in miniature: minimally modify a nominal command subject to N CBF constraints.

### Relevance for your thesis
This is the cleanest minimal example of the **add-on safety-filter architecture** — useful to cite when framing your thesis's "runtime filter wraps around an existing/nominal policy" design, before introducing OSCBF's more sophisticated task-consistent version.

---

## 4. Luo, Jakobsen, Roozing, Califano, Fang — *"Control Barrier Functions Solved with Hierarchical Quadratic Programming for Safe Physical Human-Robot Interaction"*

**Role:** Shows how to resolve conflicts between multiple CBF constraints (and task terms) via prioritization rather than a single flat QP — relevant to your planned conservatism/task-consistency analysis.

### Methodology
- Standard CBF-QP setup: h(x) defines the safe set, invariance enforced via ḣ(x) ≥ −α(h(x)), same ZCBF condition as Papers 1–2.
- Motivates **Hierarchical QP (HQP)**: when many tasks/constraints (safety *and* performance) must coexist — e.g., task-space behavior preservation, kinetic-energy limiting, null-space posture, joint limits, singularity avoidance, collision avoidance — solving them all as equally-weighted rows in one QP can degrade the *most important* task/safety objective when constraints conflict.
- HQP solves a **cascade of QPs**, one per priority level: the solution of priority-1 QP is passed as an additional equality/inequality constraint into priority-2's QP (restricting priority-2's decision variable to the null space of priority-1's active constraints), and so on — recursively, using either null-space projectors or null-space basis vectors for efficiency.
- In their pHRI setup: Priority 1 = strict safety tasks (collision avoidance, joint/torque/velocity limits, singularity avoidance) which are *never* relaxed; Priority 2 = task-space interaction-behavior preservation; Priority 3 = null-space posture — i.e., a layered soft-task hierarchy sitting *underneath* the hard safety layer.

### Key equations
- CBF definition and invariance condition (their Eqs., Section III-A) — identical ZCBF form.
- Their central methodological contribution is the **recursive HQP cascade** (Section III), not a single closed-form equation — worth summarizing as pseudocode rather than a single equation in your review.

### Relevance for your thesis
Directly useful if your conservatism analysis needs to compare "flat QP with many constraints" vs. "prioritized/hierarchical QP" — this is essentially an alternative to OSCBF's task-consistency solution (weighted single QP) via strict lexicographic prioritization instead. Good contrast case for your discussion section.

---

## 5. Morton & Pavone (2025) — *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions" (OSCBF)*

**Role:** Your most directly relevant implementation reference — already noted in memory as central to your Sprint roadmap.

### Methodology
1. **Kinematic and dynamic models.** Standard manipulator Jacobian relation ν = J(q)q̇ and joint-space/operational-space dynamics (mass matrix M(q), Coriolis c(q,q̇), gravity g(q); operational-space counterparts Λ(q), μ(q,q̇), p(q) via the dynamically-consistent inverse Jacobian J̄(q)).

2. **Task Consistency (their central conceptual contribution).** They identify that applying a CBF filter *directly to the raw control input* (joint velocity or torque) is "task-inconsistent" — it can introduce unnecessary corrective motion at the safety boundary that doesn't respect the task's operational-space/joint-space hierarchy. Three named failure modes:
   - optimizing a joint-space metric when the task lives in operational space (or vice versa),
   - ignoring a secondary null-space task,
   - optimizing torque when acceleration is the more task-relevant quantity (due to the extra inertial mapping).
   This is the "task-consistency failure mode" flagged in your memory as relevant to your conservatism analysis — it's essentially about **what the QP's cost function should be measuring**, not just what constraints it enforces.

3. **OSCBF-QP construction (kinematic / velocity-control case).** Nominal operational- and joint-space commands are computed via PD control on task error, projected through the null-space of the higher-priority task, combined into a nominal (unsafe) command, and then the CBF-QP minimally modifies this nominal *task-consistent* command (not the raw joint velocities) subject to the CBF constraint:
   ```
   min_u  Σ_i wᵢ ‖aᵢ − aᵢ_nom‖²      (task-weighted deviation, across hierarchy levels i)
   s.t.   Lf h(z) + Lg h(z) u ≥ −α(h(z))      (or HOCBF constraint if h is relative-degree 2)
          [input/torque limit constraints]
   ```
   For torque-controlled manipulators, the joint-space CBF dynamics are built from z = [q, q̇] ∈ ℝⁿ and u = Γ (torque) ∈ ℝᵐ, and the QP minimizes deviation from nominal *accelerations* (½xᵀP_QP x + qᵀQ_QP x form), consistent with their point (3) above.

4. **Slack-relaxed QP for infeasibility (their Eq. 6 — flagged as your recommended strategy).** When multiple CBF constraints conflict (common with hundreds of simultaneous constraints), the hard CBF constraint is relaxed with a slack term and large penalty ρ:
   ```
   min_{u, t}  [task-consistent objective] + ρ‖t‖²
   s.t.  Lf h(z) + Lg h(z) u + α(h(z)) ≥ −t,   t ≥ 0
   ```
   This trades strict safety guarantees for feasibility under conflicting constraints — this is the mechanism your notes flag as the recommended way to handle QP infeasibility in your G1 implementation.

5. **Relative degree handling (their Table I — flagged in your memory).** They tabulate, for each safety condition (joint position/velocity/torque limits, operational position/velocity/wrench limits, singularity avoidance, collision avoidance) whether it is naturally a relative-degree-1 (RD1) CBF or requires relative-degree-2 (RD2)/HOCBF treatment, separately for velocity-controlled vs. torque-controlled robots. This table is your direct reference for correctly formulating each G1 constraint depending on whether you drive the robot at velocity or torque level.

6. **Specific barrier functions given (Appendix):**
   - Joint velocity limit (RD1, torque control): barrier on q̇ directly.
   - Operational velocity/twist limit: barrier on ν̇(q,q̇) via manipulator kinematics.
   - Self-collision: sphere-pair model, h(q) built from forward-kinematics sphere positions x_p(q) and radii r, for each self-collision pair (j,k).
   - Dynamic obstacles: barrier inflated by relative velocity norm ‖v_rel‖ scaled by a tunable factor γ (analogous to how α tunes conservatism for static constraints) — this "velocity-inflated" obstacle barrier is a good candidate technique for your G1 dynamic-obstacle scenarios.

7. **Implementation**: CBFpy (their open-source package, Jax-based, JIT-compiled, automatic differentiation for Lie derivatives, QP solved via primal-dual interior point method), achieving multi-kHz control rates on a 7-DOF Franka arm even with 14 simultaneous constraints — directly relevant benchmark for your MuJoCo/G1 real-time performance targets.

### Key equations to cite directly
- Task hierarchy null-space projection (their Eq. ~27 area)
- OSCBF-QP objective and constraint (their "OSCBF QP," ~Eq. 36)
- Slack-relaxed QP (their Eq. 6)
- Self-collision / dynamic-obstacle barrier functions (Appendix, Eqs. 47-48 region)

### Relevance for your thesis
This is your primary implementation template. Your CBF-QP filter for the G1 should mirror: (a) task-consistent objective (minimize deviation from nominal acceleration/task command, not raw torque), (b) slack-relaxed constraints for feasibility, (c) Table-I-style relative-degree bookkeeping per constraint type, (d) sphere-based self-collision + velocity-inflated dynamic-obstacle barriers as concrete constraint templates you can adapt to the G1's 29 DOF.

---

## 6. Cai, Abanes, Evangeliou, Tzes — *"Safe Human-to-Humanoid Motion Imitation Using Control Barrier Functions"*

**Role:** Closest paper to your application domain (humanoid + CBF-QP), though the safety layer itself is simpler than OSCBF's.

### Methodology
- **Pipeline:** camera → filtered human skeleton keypoints (EMA + jump rejection) → reduced human pose (torso frame constructed from pelvis/shoulders, joint angles: torso roll/pitch, shoulder pitch/roll, elbow pitch) → affine retargeting with joint-limit projection → nominal (unsafe) joint command q_nom → **velocity-level capsule-based CBF-QP safety layer** → safe command → simulator/real robot.
- **Capsule-based collision geometry**: both the human and robot are represented as capsules (line-segment + radius primitives) for computational efficiency — this is a lighter-weight alternative to OSCBF's sphere-pair model, worth comparing in your conservatism analysis (capsules vs. spheres trade off geometric tightness against computational cost).
- **Safety layer scope**: filters only the *upper body* imitation command; the lower body is left to the robot's built-in balance controller — i.e., they explicitly decouple the safety filter from whole-body/locomotion safety, which your thesis (full 29-DOF G1, presumably including balance-relevant joints) will need to either match or extend.
- CBF-QP enforces both **self-collision avoidance** (robot-robot capsule pairs) and **human-robot collision avoidance** (human-robot capsule pairs) simultaneously, at the velocity level (i.e., h and its constraint act on q̇, making it a relative-degree-1 problem in velocity-control mode — consistent with OSCBF's Table I).
- Validated in simulation with time-series plots showing "unsafe" (nominal) vs. "safe" (filtered) joint trajectories diverging exactly when a collision distance metric approaches its safety boundary — this unsafe-vs-safe overlay plot style is a good template for presenting your own G1 CBF filter results.
- Comparative benchmarking across different collision primitives (motivating capsules over other choices) — directly relevant to a computational-cost section of your thesis.

### Key equations
The paper doesn't derive new CBF theory — it applies the standard ZCBF/CBF-QP machinery (as in Papers 1–2) with h(x) defined via capsule-to-capsule (or capsule-to-sphere) minimum distance functions for self- and human-robot collision pairs.

### Relevance for your thesis
Best available "sibling" project: humanoid + real-time CBF-QP + collision avoidance. Use it for (a) capsule-based geometry as an alternative/complement to OSCBF's spheres, (b) precedent for scoping the filter to a subset of DOF if full whole-body real-time performance becomes a bottleneck, (c) a template for result visualization (safe vs. unsafe trajectory overlays).

---

## 7. Hsu, Hu, Fisac (2024) — *"The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems"* (Annual Reviews)

**Role:** High-level taxonomy paper — useful for positioning CBF-QP filters within the broader landscape of safety-filter approaches in your introduction/background chapter, and for justifying *why* you chose CBFs over alternatives.

### Methodology (this is a review, not an original method)
- Formalizes the general **safety filter** concept: a runtime process ϕ(x,u) that monitors a proposed control action and intervenes (modifies or overrides it) to keep the system inside a maximal safe set Ω*, while otherwise passing the nominal task policy through unchanged (Proposition 1, "Perfect/Least-Restrictive Safety Filter"). This formalizes exactly the "safety-performance separation principle" that underlies every CBF-QP paper above (nominal command in, filtered safe command out).
- Every safety filter is decomposed into two functions: **monitoring** (deciding whether the current/predicted state is at risk) and **intervention** (how the action is modified).
- Taxonomy of safety filter classes, organized by monitoring type, intervention type, synthesis method, and guarantee type:
  - **Value-based / HJ reachability filters** — switch-type intervention based on a safety value function V(x), solved via the discrete-time Isaacs/Bellman equation; scales poorly (state dim ≲ 6) but gives the *maximal* safe set.
  - **Control Barrier Functions (Sec. 3.2)** — your thesis's chosen class; framed here as a smaller, more scalable (but generally more conservative) safe set Ω ⊂ Ω* compared to the reachability-based maximal set, obtained via optimization-based intervention (the CBF-QP).
  - **Rollout-based filters** (model predictive shielding, forward-reachable sets, tube MPC) — check safety by forward-simulating a fallback policy; more general but computationally heavier than CBFs.
  - **Data-driven filters** — learned safety critics / learned CBFs from data, useful when analytic h(x) is hard to construct, but lose formal guarantees.
  - **Multi-agent / probabilistic extensions** — relevant if you later add human-in-the-loop or dynamic-obstacle uncertainty to your G1 scenario.
- Notes the key CBF *limitation* that motivates other approaches: while CBF-type filters are efficient to *evaluate*, finding a valid CBF for a given system remains a nontrivial synthesis problem, and CBF-based methods currently lack general constructive mechanisms (unlike HJ reachability, which is fully constructive but computationally intractable at scale).

### Key equations
Not a source of equations you'll cite directly in your methods chapter — rather a source of precise terminology (safe set Ω, maximal safe set Ω*, safety monitor, controlled-invariant set, least-restrictive filter) useful for the theoretical framing of your introduction/background chapter, and Proposition 1 (Perfect Safety Filter) as the formal justification for the safety-performance separation your whole thesis architecture assumes.

### Relevance for your thesis
Use to (a) justify choosing CBF-QP over reachability/MPC-based filters (real-time feasibility on a 29-DOF humanoid rules out HJ reachability and makes rollout-based MPC expensive), (b) give your thesis a rigorous vocabulary for "safety filter," "safe set," "conservatism" (the gap between your CBF's Ω and the true maximal safe set Ω* is precisely what your planned conservatism analysis should quantify).

---

## 8. "Safety and Efficiency in Robotics" (IEEE Xplore magazine article)

**Role:** Short tutorial-style piece on CBFs for industrial/collaborative robotics.

### Methodology
- Restates the core CBF idea (forward invariance of a safe set achieved via the CBF's defining inequality) in an applied, less formal register aimed at practitioners.
- Focuses on manipulator-relevant applications: human-robot collaboration in industrial settings, with an emphasis on practical/robust design methodology and experimental validation in a real collaborative-robotics industrial setup.
- References the standard CBF literature (Ames et al.) as its theoretical basis, without introducing new formalism.

### Relevance for your thesis
Useful as a secondary/tertiary citation for motivating industrial relevance of CBF-based safety in your introduction, but not a primary technical source — the Ames et al. papers and OSCBF remain your primary equations sources.

---

## 9. CBFpy (StanfordASL / Daniel Morton) — GitHub repository

**Role:** Reference software implementation accompanying OSCBF.

### What it is
An open-source Python/Jax package for constructing and solving CBF-QPs: automatic differentiation for Lie derivatives, JIT compilation to XLA, and a Jax-based primal-dual interior-point QP solver — enabling multi-kHz control rates even with many simultaneous constraints (as reported in the OSCBF paper).

### Relevance for your thesis
Candidate library (or reference architecture to reimplement/adapt in your MuJoCo + Python environment) for solving the CBF-QP at each control step for the G1. Worth citing in your implementation chapter as either a tool you build on or as the design pattern (Jax autodiff + JIT + interior-point QP) you replicate for your own solver.

---

## Cross-Paper Synthesis Table

| Paper | Safety filter type | Constraint level | Handles multiple constraints? | Task-consistency addressed? | Infeasibility handling |
|---|---|---|---|---|---|
| Ames et al. 2017 | CBF-CLF-QP | Single scalar h(x), rel. degree 1 or r | No (single constraint focus) | No | Slack on CLF only |
| Ames et al. 2019 | Survey of above | — | — | — | — |
| Rauscher et al. 2016 | CBF-QP add-on to nominal controller | Multiple h_i(x) | Yes (N stacked constraints) | No | Not addressed |
| Luo et al. (HQP) | Hierarchical CBF-QP cascade | Multiple, prioritized | Yes, via strict priority levels | Implicitly (via priority) | Lower-priority tasks yield to higher |
| Morton & Pavone (OSCBF) | Task-consistent CBF-QP | Many (100s), RD1/RD2 per Table I | Yes | **Yes — central contribution** | Slack-relaxed QP (Eq. 6) |
| Cai et al. (humanoid imitation) | Capsule-based CBF-QP | Velocity-level, RD1 | Yes (self + human-robot pairs) | No | Not discussed |
| Hsu, Hu, Fisac 2024 | Taxonomy (all of the above are "optimization-type intervention" filters) | — | — | — | — |

---

## Suggested Use in Your Thesis
- **Chapter 2 (Background/Theory):** Ames et al. 2017 (rigor) + Ames et al. 2019 (pedagogy) + Hsu/Hu/Fisac (taxonomy/vocabulary).
- **Chapter 3 (Related Work — manipulator/humanoid CBFs):** Rauscher et al., Luo et al., Cai et al., "Safety and Efficiency in Robotics."
- **Chapter 4 (Methodology — your CBF-QP design for G1):** OSCBF as primary template (task-consistent objective, slack-relaxed QP, Table I relative-degree bookkeeping, sphere/capsule collision barriers); CBFpy as implementation reference.
- **Chapter 6 (Conservatism/discussion):** OSCBF's task-consistency failure mode, Luo et al.'s hierarchical alternative, and Hsu/Hu/Fisac's Ω vs. Ω* framing as your analytical lens.

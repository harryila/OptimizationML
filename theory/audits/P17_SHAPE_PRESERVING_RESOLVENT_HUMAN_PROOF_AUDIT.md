# P17 shape-preserving resolvent human proof audit

Status: **unsigned and pending independent human review**.

This packet is a checklist, not evidence that review occurred. A reviewer
should sign only after reconstructing the indicated arguments independently
of the generator and committed JSON artifact.

## Locked object

- Matrix domain: every fixed finite real rectangular matrix space.
- Arithmetic: exact real arithmetic for theorem claims.
- Normalization:
  `U/(||U||_F+epsilon)`, with `epsilon=1/10000000` outside the norm.
- Jordan stage:
  `q(x)=(6889/2000)x-(191/40)x^3+(4063/2000)x^5`, composed five times.
- Formula provenance: KellerJordan/Muon revision
  `f98f1cacc0263b04290753e32be8d498c1efc806`.
- P13 map: `A_epsilon=E_h,epsilon+G_epsilon`.
- P14 graph:
  `B=A_epsilon+1000I`, `J=(I+B/1000)^(-1)`, `Y=1000(I-J)`.
- Shape channel: `X(S)=E_h,epsilon(J(S))`.
- Gate variable: `q=||S||_F^2`.
- Gate: zero for `q<=1/4`, cap times
  `6t^5-15t^4+10t^3` on `1/4<q<1`, and the cap for `q>=1`.
- Interface: `T=(1-theta)Y/c+theta X`.
- Locked designs:
  - primary: cap `3/4`, `c=4096`, `eta=1/128000`;
  - full-step: cap `1/8`, `c=8192`, `eta=1/32000`.

The upstream revision contains the additive-epsilon Jordan formula, but does
not contain P13's repair, the `1000I` shunt, the resolvent, or this gate.

## A. Common singular basis and sign

- [ ] Recheck C22's equivariance/stabilizer argument and verify that `S`,
  `J(S)`, `Y(S)`, and `X(S)` admit a common singular-vector representation.
- [ ] Verify the statement at repeated singular values is basis-independent.
- [ ] Verify zero singular values stay zero and rectangular null-side
  components vanish.
- [ ] For one Jordan stage, verify the factor in `x^2` has positive leading
  coefficient and negative discriminant; conclude all five stages preserve
  nonnegative singular modes.
- [ ] Confirm that this sign argument is for the exact-real spectral map and
  does not silently assert continuity of a backend-specific BF16 program.

## B. Dimension-uniform raw-shape gain

In a positive mode, independently derive

\[
\sigma_i=[1+\lambda(\mu+g)]x_i+\lambda e_i,
\qquad
\frac{e_i}{\sigma_i}
=\frac{\lambda^{-1}H_i}{\lambda^{-1}+\mu+g+H_i}.
\]

- [ ] From the certified derivative upper bound and `h(0)=0`, verify
  `0<=h(w)/w<=4848763/10000` on `[0,1]`.
- [ ] Verify `H_i<=b/[epsilon(1+z)]` and
  `g>=d_hat(z)/epsilon`.
- [ ] Prove `(1+z)d_hat(z)>=U_0` on the constant branch.
- [ ] On the tail, expand `(1+z)d_hat(z)` as a convex combination of
  `199437/1250` and `41528474059081/260261360000`; verify both exceed
  `U_0=6602082433275499863/41641817600000000`.
- [ ] Check monotonicity of `H/(lambda^-1+mu+g+H)` in `H` and `g`.
- [ ] Reduce and independently match

  \[
  M_X=\frac{20191130443162880000000}
            {26793221204801899863}.
  \]

- [ ] Confirm the lift from modal inequalities to Frobenius norm is valid in
  arbitrary finite rank and does not use sampled spectra.

## C. Gate and pointwise sector

- [ ] Differentiate the quintic transition twice and verify value, first
  derivative, and second derivative match at both joins.
- [ ] Verify the maximum derivative with respect to `q` is
  `(5/2) theta_bar`.
- [ ] Verify `T(0)=0`.
- [ ] Starting from the singular graph, put
  `k_i=mu+g+H_i>=mu=1000`, derive
  `Y_i/S_i=k_i/(1+lambda*k_i)`, and hence independently verify
  `500<=Y_i/S_i<1000` without treating C20's nonsymmetric incremental IQC as
  a modal or Loewner bound. Then derive

  \[
  m=(1-\bar\theta)500/c,
  \qquad
  M=(1-\bar\theta)1000/c+\bar\theta M_X.
  \]

- [ ] Check `M_X>1000/c` for both locked designs.
- [ ] Verify modewise interval inclusion implies
  `||T(S)-gamma S||_F<=K||S||_F` with midpoint `gamma` and radius `K`.
- [ ] Confirm this is an origin-centered **pointwise** supply. It is not an
  incremental sector, monotonicity theorem, or global derivative bound.
- [ ] Differentiate the gate and identify the omitted rank-one term before
  accepting the non-incremental qualification.

## D. Exact smooth-PL LMIs

For each locked design:

- [ ] Rebuild the normalized EMA/Nesterov dynamics at `beta=19/20`.
- [ ] Rebuild both directed `(-1,1)` smooth interpolation matrices.
- [ ] Rebuild the next-point PL supply for smoothness `10` and PL constant
  `1`.
- [ ] Verify the P6 proof uses only the pointwise residual supply and never
  differentiates `T` or compares two interface inputs.
- [ ] Verify every multiplier is nonnegative.
- [ ] Verify objective-value coefficients cancel exactly.
- [ ] Recompute every leading principal minor of the storage matrix.
- [ ] Recompute every leading principal minor of the negative `4 x 4` LMI.
- [ ] Confirm strict positivity using exact rational arithmetic, not decimal
  eigensolver output.
- [ ] Verify the rates are exactly:
  - primary: `(16777215/16777216)^2` at `eta=1/128000`;
  - full-step: `(16777209/16777216)^2` at `eta=1/32000`.
- [ ] Check the conclusions: objective gap, gradient, and momentum decay;
  summable steps; convergence to some global minimizer. Do not infer a unique
  minimizer or arbitrary-pair contraction from PL.

## E. Unsafe derivative and gate negative control

- [ ] Rebuild the exact rational witness at `t=63/10000`.
- [ ] Verify `19/10000<S<1/500` and
  `-147000<X'(S)<-146000` exactly.
- [ ] Independently run the Arb recurrence on the committed complete interval
  and verify the forward graph derivative is positive while `X'` is strictly
  negative throughout.
- [ ] Confirm both locked gates are exactly zero at the unsafe witness.
- [ ] Rebuild the under-sized passive-region control and verify its gate is
  already on the raw-shape plateau.
- [ ] Verify `-19000<T'(S)<-18000` and the second Jury margin is negative.
- [ ] Confirm this is a local/incremental control at a non-equilibrium signal,
  not a proof that an unbiased smooth-PL trajectory diverges.

## F. Fidelity and computation evidence

- [ ] Verify the P16 gates were retained unchanged:
  `chi>=1/1000` and shaping retention at least `1/10`.
- [ ] Independently recompute the graph residual for the frozen binary64
  candidate on `diag(3,4)`.
- [ ] Inflate the root by `||r||/2` and the Yosida output by `500||r||` before
  making either fidelity decision.
- [ ] Recompute the raw-shape channel over the root enclosure using outward
  rounding.
- [ ] Verify both canonical gate decisions are interval-separated from their
  thresholds.
- [ ] Inspect the declared spectrum-grid and realistic-rank diagnostics,
  including failures and rank accumulation.
- [ ] Confirm those grids are numerical evidence, not certified global
  fidelity extrema.
- [ ] Confirm every declared FP64 solver call fails closed on its computed P15
  residual, while recognizing that P17 has not yet propagated solver error
  through `X` or certified rounded residual evaluation.

## G. Scope and provenance

- [ ] Confirm every claim states the matrix domain, normalization, epsilon,
  polynomial coefficients, and five-stage count.
- [ ] Confirm the exact generator and independent reconstruction agree on all
  authoritative rational fields.
- [ ] Verify the committed source hashes, source commit, artifact commit, and
  predecessor P16 artifact/tag references.
- [ ] Confirm the study records seed, hardware, software, and Git SHA.
- [ ] Confirm no claim covers BF16, FP32, weight decay, aspect scaling,
  stochastic gradients, literal upstream Muon, or neural-network training.

## Reviewer sign-off

- Reviewer name:
- Affiliation or role:
- Date:
- Commit reviewed:
- Artifact SHA-256:
- Independent reconstruction command/output:
- Exceptions or requested corrections:
- Signature or verifiable approval reference:

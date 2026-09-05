# P16 equivariant resolvent-solver human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record.  Numerical solver tests, interval evaluation, exact
artifact replay, and independent code reconstruction do not constitute a
human proof audit.

The theorem under review is `theory/equivariant_resolvent_solver.md`, and the
theorem-facing implementation is
`src/passive_muon/equivariant_resolvent_solver.py`.  The canonical artifact,
generator, and independent reconstruction are respectively
`results/summaries/equivariant_resolvent_solver_certificate.json`,
`scripts/certify_equivariant_resolvent_solver.py`, and
`scripts/reconstruct_equivariant_resolvent_solver.py`.

## Fixed mathematical object

For every fixed finite real rectangular matrix space and every `epsilon>0`,
review P13's monotone operator

\[
A_\epsilon=E_{h,\epsilon}+G_\epsilon,
\]

P14's shift and resolvent

\[
B=A_\epsilon+1000I,
\qquad
J=(I+B/1000)^{-1},
\]

and the output `Y=1000(I-J)`.  The five Jordan stages use coefficients
`6889/2000`, `-191/40`, and `4063/2000`, after normalization by
`M/(||M||_F+epsilon)`.

P16 reduces the exact resolvent to singular values, solves its
diagonal-plus-rank-one equations by safeguarded Newton, reconstructs the FP64
matrix candidate, and reevaluates `B` on that stored candidate with a second
SVD before applying P15's literal graph-residual test. Solver correctness and
Muon fidelity are separate claims.

## Reviewer checklist

### 1. Inherited P13--P15 premises

- [ ] Confirm P13 proves that `A_epsilon` is continuous, full-domain,
  zero-preserving, and monotone on every finite rectangular Frobenius space.
- [ ] Confirm P14's `B=A_epsilon+1000I` is `1000`-strongly monotone and
  `F=I+B/1000` is `2`-strongly monotone and bijective.
- [ ] Confirm `J=F^{-1}` is single-valued and `Lip(J)<=1/2`.
- [ ] Confirm P15 deploys `Y_hat=1000*(S-U_hat)`, not `B(U_hat)`, and checks
  `||S-U_hat-B(U_hat)/1000||_F<=||S||_F/250+rbar`.
- [ ] Check that pending human review of inherited results remains disclosed.

Reviewer notes:

> Pending.

### 2. Bi-orthogonal equivariance

- [ ] Prove Frobenius additive normalization commutes with every action
  `M -> Q*M*R^T`, for `Q in O(m)` and `R in O(n)`.
- [ ] Verify each rectangular quintic stage is bi-orthogonally equivariant,
  including the wide/tall implementation convention.
- [ ] Verify P13's radial `G_epsilon` and the identity shunt are equivariant.
- [ ] Derive `F(QMR^T)=QF(M)R^T`.
- [ ] Use uniqueness, rather than an unproved selection convention, to derive
  `J(QSR^T)=QJ(S)R^T` and the corresponding identity for `Y`.

Reviewer notes:

> Pending.

### 3. Singular-vector preservation at multiplicities and zeros

- [ ] Apply the complete stabilizer of a rectangular diagonal SVD input.
- [ ] Check that simultaneous rotations on each repeated positive block force
  a scalar identity block in the resolvent.
- [ ] Check that independent left/right rotations on null spaces force the
  complete null block and its cross terms to zero.
- [ ] Check sign changes eliminate cross terms between distinct positive
  blocks.
- [ ] Verify the stage sign argument: the factor in `x^2` has discriminant
  `-2594691/500000<0` and positive leading coefficient.
- [ ] From positivity of `p` and the shunt, prove `x_i>=0` and
  `x_i=0 iff sigma_i=0`.
- [ ] Confirm reconstruction is independent of arbitrary SVD bases inside
  repeated and zero subspaces.

Reviewer notes:

> Pending.

### 4. Coupled scalar equations

- [ ] Starting from `F(U_hat)=S`, derive exactly
  `Phi_i=(1+lambda*(mu+p(z)/r))*x_i+lambda*h(x_i/(r+epsilon))-sigma_i`.
- [ ] Confirm `r=||x||_2`, `z=r/epsilon`, and that multiplicities are counted
  in this Euclidean norm.
- [ ] Confirm the only coupling is through `r`.
- [ ] Verify at an approximate root that the deployed output singular values
  remain `(sigma_i-x_i)/lambda`.
- [ ] Prove the Frobenius graph-residual norm equals `||Phi(x)||_2` exactly.
- [ ] Confirm no sampled diagonal calculation is substituted for the original
  full-matrix theorem.

Reviewer notes:

> Pending.

### 5. Rank-one Jacobian algebra

- [ ] Re-derive
  `c'(r)=(z*d_hat(z)-p(z))/(epsilon^2*z^2)`.
- [ ] Differentiate `w_i=x_i/(r+epsilon)` including its radial term.
- [ ] Verify every sign and power of `epsilon` in
  `D_ii=1+lambda*c+lambda*h'(w_i)/(r+epsilon)` and
  `a_i=lambda*x_i*(c'-h'(w_i)/(r+epsilon)^2)`.
- [ ] Confirm `D Phi=diag(D)+a*v^T`, with `v=x/r`, even though this Jacobian
  need not be symmetric.
- [ ] Reconstruct the Sherman--Morrison solve and its denominator exactly.
- [ ] Check its scalar-operation cost is `O(k)` after response evaluation.

Reviewer notes:

> Pending.

### 6. Positivity of the Sherman--Morrison factors

- [ ] Verify the three exact early-band margins displayed in (P16.25).
- [ ] On the tail, derive
  `d_hat+a/(1+z)=(a-Gamma)z/(1+z)^2>=0` and the exact positive value of
  `a-Gamma`.
- [ ] Conclude `D_ii>=1+lambda*mu=2` for every coordinate and radius.
- [ ] Independently derive `Sym(D Phi)>=2I` from full-matrix monotonicity.
- [ ] Verify that positive-definite symmetric part implies positive
  determinant for a real matrix.
- [ ] Apply the determinant lemma to prove the Sherman--Morrison denominator
  is strictly positive.
- [ ] Ensure the implementation still rejects a nonpositive computed
  denominator rather than assuming away rounding failure.

Reviewer notes:

> Pending.

### 7. Nonsmooth-looking branches

- [ ] Verify the exact zero-input output and derivative branch.
- [ ] For `0<z<=z0`, verify `p/r=U0/epsilon` and `c'(r)=0`, avoiding `0/0`.
- [ ] At `z=z0`, verify branch continuity, equality of first derivatives, and
  `z*d_hat-p=0`.
- [ ] Confirm only a second-derivative kink remains at the switch.
- [ ] Verify individual zero coordinates at nonzero radius cause no division
  and decouple from the rank-one term.
- [ ] Confirm repeated singular values require no derivative of SVD vectors.
- [ ] Verify the fixed-radius scalar derivative is strictly positive and hence
  preserves singular-value ordering.

Reviewer notes:

> Pending.

### 8. Global safeguarded-Newton theorem

- [ ] Confirm the reduced residual is `C1` with locally Lipschitz Jacobian at
  zero, across `z0`, and elsewhere.
- [ ] Verify `||D Phi^{-1}||_2<=1/2` follows from its symmetric part, without
  assuming the Jacobian is symmetric.
- [ ] Derive the exact descent identity
  `grad(||Phi||^2/2)^T*d=-||Phi||^2` for the Newton direction.
- [ ] Verify the stated Armijo condition and backtracking parameters.
- [ ] Prove the initial merit level set is compact using strong monotonicity.
- [ ] Establish uniform boundedness/Lipschitzness on that level set and a
  bounded one-step trial enlargement, then establish a positive lower bound
  on accepted step sizes.
- [ ] Conclude residual convergence and convergence to the unique root.
- [ ] Confirm finite stopping follows for every positive P15 threshold, while
  `S=0` is handled exactly.
- [ ] Distinguish reliable exact-arithmetic termination from a uniform useful
  iteration bound and from a certified FP64 result.

Reviewer notes:

> Pending.

### 9. FP64 success/failure contract

- [ ] Verify the implementation uses damping/backtracking, finite iteration
  and backtracking caps, and nonfinite-value checks.
- [ ] Check zero input, near-zero input, repeated singular values, individual
  zeros, and the P13 switch are all exercised.
- [ ] Confirm every declared successful call recomputes the actual or exactly
  equivalent singular-coordinate P15 residual.
- [ ] Confirm calls that miss the threshold return an explicit failure and no
  optimizer output.
- [ ] Verify successful output is exactly `1000*(S-U_hat)`.
- [ ] Confirm `rbar_fp64` is not called rigorous until a rounding envelope is
  supplied in P18.

Reviewer notes:

> Pending.

### 10. Algebraic spectral-shaping witness

- [ ] Check the canonical input, additive epsilon, coefficients, stage count,
  lambda, mu, solver tolerance, and precision are all locked.
- [ ] Use outward-rounded intervals to enclose both output singular gains.
- [ ] Verify the two gain intervals are disjoint.
- [ ] Confirm this proves only algebraic noncollapse and is not by itself
  called meaningful Muon fidelity.
- [ ] Verify repeated and zero singular values are not used to manufacture a
  spurious gain distinction.

Reviewer notes:

> Pending.

### 11. Meaningful-fidelity decision rule

- [ ] Recompute the best scalar coefficient
  `alpha_T=<T,S>/||S||^2` and normalized residual
  `chi(T;S)=||T-alpha_T*S||/||T||`.
- [ ] Verify `chi` is invariant to global nonzero output scaling.
- [ ] On the pre-existing `diag(3,4)` comparator, independently compute the
  exact-real five-stage additive-epsilon Jordan reference.
- [ ] Reconstruct `R_shape=chi(Y;S)/chi(T_Muon;S)` with directed intervals.
- [ ] Apply the frozen pre-certificate `R_shape>=1/10` gate without post hoc weakening.
- [ ] Report distances to `500*S`, `750*S`, `1000*S`, the best scalar fit,
  and the two-mode gain spread.
- [ ] If only distinct gains pass, classify the selected point as “distinct
  but effectively scalar” and require a `(lambda,mu)` frontier.

Reviewer notes:

> Pending.

### 12. Cost and empirical solver evidence

- [ ] Separate one-time SVD and reconstruction cost from `O(k)` work per
  Newton iteration.
- [ ] Compare against the repository's locked count of fifteen dense matrix
  multiplications for five Jordan quintic stages.
- [ ] Report iteration and backtracking distributions, not only averages.
- [ ] Report the worst attained residual and its threshold slack.
- [ ] Include zero, near-zero, repeated, rank-deficient, random-orientation,
  and P11-guard-scale cases.
- [ ] Include all declared solver failures and negative controls.
- [ ] Confirm sampling is treated as implementation falsification and cost
  evidence, never as proof of the global theorem.

Reviewer notes:

> Pending.

### 13. Reconstruction, provenance, and scope

- [ ] Run the primary generator and independently implemented reconstruction.
- [ ] Confirm exact fields and source hashes match the committed artifact.
- [ ] Verify the branch source commit, artifact commit, artifact hash, and
  annotated checkpoint tag are distinguished.
- [ ] Confirm seeds, hardware, software, FP64 backend, and Git SHA are recorded
  for numerical evidence.
- [ ] Confirm P16 does not claim certified FP64 rounding, BF16, accelerator
  parity, stochastic-gradient robustness, weight decay, aspect scaling, or
  neural-network convergence.
- [ ] Confirm the result is called a certified Muon-derived reference
  architecture, not globally stable literal upstream Muon.

Reviewer notes:

> Pending.

## Sign-off

Reviewer name:

> Pending.

Affiliation or contact:

> Pending.

Date:

> Pending.

Decision and required corrections:

> Pending.

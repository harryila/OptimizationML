# Floored-normalizer certificate (artifact note)

This is a proof and result record for the repository, not a paper draft.  It
uses real arithmetic throughout.

## Statement and scope

Fix any finite matrix shape `m x n`, equip it with the Frobenius inner
product, and let `h` be the odd polynomial obtained by five repetitions of

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3+\frac{4063}{2000}s^5.
\]

Let `H_h` be the corresponding rectangular singular-value map and, for
`c>0`, define

\[
N_c(M)=\frac{M}{\max\{c,\lVert M\rVert_F\}},
\qquad F_{h,c}=\mathcal H_h\circ N_c.
\]

The authoritative global deficit is the pairwise quantity

\[
\delta(F)=\sup_{A\ne B}
\left[-\frac{\langle F(A)-F(B),A-B\rangle_F}
{\lVert A-B\rVert_F^2}\right]_+.
\]

The committed machine certificate establishes

\[
159.549525785
<\delta(F_{h,1})\le
\frac{41528474059081}{260261360000}
=159.564501081070\ldots.
\]

The upper endpoint is uniform over every finite matrix shape.  It is a
certified sufficient conductance, not the exact minimum for each fixed shape.
The bracket has relative width `0.009386%` measured from the lower witness.

## 1. Full rectangular tangent reduction

Put

\[
a=\min_{0\le s\le1}h'(s),\qquad
b=\max_{0\le s\le1}h'(s).
\]

At a matrix `Z` with singular values `sigma_i` in `[0,1]`, the derivative
`A_Z=D H_h(Z)` is Frobenius-self-adjoint.  In the standard singular-vector
tangent decomposition, its eigenvalues are

\[
h'(\sigma_i),\qquad
\frac{h(\sigma_i)-h(\sigma_j)}{\sigma_i-\sigma_j},\qquad
\frac{h(\sigma_i)+h(\sigma_j)}{\sigma_i+\sigma_j},\qquad
\frac{h(\sigma_i)}{\sigma_i},
\]

with the continuous values used at ties and zero.  The final family is the
rectangular null-side mode.  Since `h` is odd, every displayed quantity is a
secant average of the even function `h'` on `[-1,1]`.  Therefore, on the
entire rectangular tangent space, including repeated and zero singular
values,

\[
aI\preceq A_Z\preceq bI.
\]

This step is what promotes the scalar interval calculation to a full-matrix
certificate; no diagonal-only repair claim is being made.

## 2. Projection/anticommutator lemma

Let `A=A*`, `aI <= A <= bI`, and let `Q` be any orthogonal projection.  For a
unit vector `v`, set `beta=||Qv||`.  The self-adjoint rank-two operator

\[
\operatorname{Sym}\bigl((Qv)\otimes v\bigr)
\]

has its only nonzero eigenvalues

\[
\frac{\beta(\beta+1)}2,
\qquad
\frac{\beta(\beta-1)}2.
\]

Bounding `A` by `aI` on the positive eigenspace and by `bI` on the negative
eigenspace gives

\[
\left\langle v,\frac{AQ+QA}{2}v\right\rangle
\ge
\frac{(a+b)\beta^2+(a-b)\beta}{2}.
\]

For the certified Jordan slope range, `a+b>0` and `b+3a>0`; minimizing the
right-hand side on `0<=beta<=1` gives

\[
\Gamma(a,b)=-\frac{(b-a)^2}{8(a+b)}.
\]

Thus

\[
\lambda_{\min}\!\left(\frac{AQ+QA}{2}\right)\ge\Gamma(a,b)
\]

in every finite dimension.  The bound is sharp for unrestricted pairs
`(A,Q)`.  That generic sharpness does not imply equality for the linked
operator `A=D H_h(Z)` at a particular fixed matrix shape.

## 3. Floor, boundary, and pairwise inequality

For `c=1`, inside the unit ball the derivative of `F` is `A_Z`.  At a point
`M=rZ` outside the ball, where `r>1` and `||Z||_F=1`,

\[
DN_1(M)=\frac{Q_Z}{r},\qquad
Q_Z=I-Z\otimes Z,
\]

and hence

\[
\operatorname{Sym}DF(M)=\frac{A_ZQ_Z+Q_ZA_Z}{2r}
\succeq \Gamma(a,b)I.
\]

The last implication uses `Gamma(a,b)<0` and `r>1`.  At the switching sphere
there is no ordinary derivative.  The pairwise proof avoids inventing one:
`N_1` is projection onto the closed Frobenius unit ball and is nonexpansive;
`H_h` has bounded derivative on that ball; therefore `F` is globally
Lipschitz.  On any line segment, `F` is absolutely continuous, the line meets
the switching sphere on a measure-zero set, and integration of the almost-
everywhere derivative gives

\[
\langle F(Y)-F(X),Y-X\rangle_F
\ge \Gamma(a,b)\lVert Y-X\rVert_F^2.
\]

As a supplementary boundary check,

\[
\partial_C N_1(Z)=
\{I-\theta Z\otimes Z:0\le\theta\le1\},
\qquad \lVert Z\rVert_F=1.
\]

The symmetric composite elements interpolate between `A_Z` and
`(A_Z Q_Z+Q_Z A_Z)/2`, so the same lower bound holds throughout the Clarke
hull.

## 4. Rigorous scalar enclosure and exact witness

The executable certificate evaluates the five-stage recurrences

\[
x_{k+1}=q(x_k),\qquad
d_{k+1}=q'(x_k)d_k
\]

with outward-rounded Arb balls.  It adaptively bisects dyadic intervals until
every leaf proves

\[
-159.5496<h'(s)<484.8763
\qquad (0\le s\le1).
\]

Both a 160-bit and a 224-bit replay accept the same 25,370-leaf cover, with
maximum dyadic depth 32.  Substitution of the exact rational endpoints into
the projection lemma gives

\[
\bar\delta_1
=\frac{41528474059081}{260261360000}
=159.564501081070\ldots.
\]

For a pairwise lower witness, embed the scalar inputs

\[
s_-=0.006349576126,
\qquad s_+=0.006349578126
\]

as a rank-one matrix mode.  Both lie strictly inside the unit floor, so the
normalizer is linear there.  Exact rational evaluation of the finite pair
proves

\[
-\frac{h(s_+)-h(s_-)}{s_+-s_-}
>159.549525785.
\]

The exact ratio is large, so the manifest records its canonical SHA-256 and a
machine-checked rational lower bound rather than embedding a 39,000-digit
fraction.

## 5. Scaling, repair, and simplified stability corollary

Since

\[
F_{h,c}(cX)=F_{h,1}(X),
\]

changing variables in the unrestricted pairwise supremum gives the exact law

\[
\delta(F_{h,c})=\frac{\delta(F_{h,1})}{c}.
\]

Consequently

\[
M\longmapsto F_{h,c}(M)+\frac{\bar\delta_1}{c}M
\]

is globally monotone for every finite matrix shape.  Adding any `mu>0`,

\[
\rho=\frac{\bar\delta_1}{c}+\mu,
\]

makes the repaired map `mu`-strongly monotone.  Its zero-input gain is
`(6889/2000)^5=484.876287100182...`, so the certified correction at `c=1` is
about `32.91%` of that differential gain, not the roughly `484.88` generic
Lipschitz correction.

For the explicitly simplified continuous-time system

\[
\dot M=-\bigl(F_{h,c}(M)+\rho M-b\bigr),
\]

global Lipschitzness gives unique trajectories.  Strong monotonicity and
coercivity give one equilibrium for each constant `b`.  For two trajectories,

\[
\frac12\frac{d}{dt}\lVert M_1-M_2\rVert_F^2
\le-\mu\lVert M_1-M_2\rVert_F^2,
\]

so

\[
\lVert M_1(t)-M_2(t)\rVert_F
\le e^{-\mu t}\lVert M_1(0)-M_2(0)\rVert_F.
\]

This is a continuous-time integrator-feedback/interconnection result.  It is
not a theorem about momentum Muon training, discrete learning-rate stability,
or a BF16 implementation.

## Reproduce

```bash
uv run --locked python scripts/certify_floored_repair.py \
  --output results/summaries/floored_repair_certificate.json
```

# Additive-epsilon deficit certificate (artifact note)

This is a proof and result record for the repository, not a paper draft. It
uses exact real arithmetic except where it explicitly describes the separate,
pinned BF16 implementation.

## Statement and scope

Fix a finite matrix shape `m x n`, equip it with the Frobenius inner product,
and let `h` be the odd polynomial obtained by five repetitions of

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5.
\]

Let `H_h` be the corresponding rectangular singular-value map. For every
real `epsilon>0`, define the exact-real additive-normalized operator

\[
N_\epsilon(M)=\frac{M}{\lVert M\rVert_F+\epsilon},
\qquad
E_{h,\epsilon}=\mathcal H_h\circ N_\epsilon.
\tag{A.1}
\]

For each fixed shape, its authoritative global monotonicity deficit is

\[
\delta_{m,n}(E)=\sup_{X\ne Y}
\left[-\frac{\langle E(X)-E(Y),X-Y\rangle_F}
{\lVert X-Y\rVert_F^2}\right]_+.
\tag{A.2}
\]

The certificate proves, for every finite positive `m,n`,

\[
\delta_{m,n}(E_{h,\epsilon})
\le
\frac{1}{\epsilon}
\frac{6602082433275499863}{41641817600000000}
=\frac{158.544530805386\ldots}{\epsilon}.
\tag{A.3}
\]

For `2 x 2`, and hence for every shape with `min(m,n)>=2` by zero-padding,
an exact rational finite pair at `epsilon=1`, scaled exactly for general real
`epsilon>0`, proves

\[
\delta_{m,n}(E_{h,\epsilon})
>
\frac{1}{\epsilon}\frac{98823281}{625000}
=\frac{158.1172496}{\epsilon}.
\tag{A.4}
\]

Thus the dimension-uniform upper endpoint is within about `0.270231%` of an
explicit lower witness available in every shape of rank at least two. It is a
sufficient global repair, not a claim of the exact deficit or minimal repair
for any fixed shape.

## 1. Exact epsilon scaling

For every matrix `X`,

\[
N_\epsilon(\epsilon X)
=\frac{\epsilon X}{\epsilon\lVert X\rVert_F+\epsilon}
=N_1(X),
\qquad
E_{h,\epsilon}(\epsilon X)=E_{h,1}(X).
\tag{A.5}
\]

The change of variables `M=epsilon X` is a bijection on each fixed matrix
space. The numerator in (A.2) gains one factor of `epsilon`, while the squared
denominator gains two. Taking the same unrestricted supremum in the new
variables gives the equality

\[
\boxed{
\delta_{m,n}(E_{h,\epsilon})
=\frac{\delta_{m,n}(E_{h,1})}{\epsilon}}
\qquad(\epsilon>0).
\tag{A.6}
\]

This identity is for the unrestricted global deficit. A restricted domain
must be scaled along with `epsilon`.

## 2. The origin and the additive normalizer derivative

Write `r=||M||_F`. Although the norm alone is not differentiable at zero, the
quotient in (A.1) is. Indeed,

\[
\left\|N_\epsilon(M)-\frac{M}{\epsilon}\right\|_F
=\frac{r^2}{\epsilon(\epsilon+r)}=o(r).
\tag{A.7}
\]

Since the five-stage polynomial map has derivative
`(6889/2000)^5 I` at zero,

\[
DE_{h,\epsilon}(0)
=\frac1\epsilon
\left(\frac{6889}{2000}\right)^5 I
=\frac1\epsilon
\frac{15516041187205853449}{32000000000000000}I
\succ0.
\tag{A.8}
\]

Thus the origin is not a hidden nondifferentiable boundary or a negative
mode.

At `M!=0`, put

\[
U=\frac{M}{r},\qquad
t=\frac{r}{r+\epsilon}\in(0,1),\qquad
P=U\otimes U,\qquad Q=I-P,
\tag{A.9}
\]

where `P` and `Q` act on the Frobenius matrix space. Direct differentiation
gives

\[
DN_\epsilon(M)
=\frac{1-t}{\epsilon}(I-tP)
=\frac{1-t}{\epsilon}\bigl((1-t)I+tQ\bigr).
\tag{A.10}
\]

The formula tends continuously to `I/epsilon` as `M` tends to zero. Hence
`N_epsilon`, and therefore `E_(h,epsilon)`, is globally continuously
differentiable.

## 3. Full rectangular spectral derivative

Set `Z=N_epsilon(M)` and `A_Z=D H_h(Z)`. The operator `A_Z` is
Frobenius-self-adjoint. If the singular values of `Z` are `sigma_i`, its
standard full rectangular tangent decomposition has the modes

\[
h'(\sigma_i),\qquad
\frac{h(\sigma_i)-h(\sigma_j)}{\sigma_i-\sigma_j},\qquad
\frac{h(\sigma_i)+h(\sigma_j)}{\sigma_i+\sigma_j},\qquad
\frac{h(\sigma_i)}{\sigma_i}.
\tag{A.11}
\]

Continuous limiting values are used at repeated or zero singular values. The
last family contains the rectangular null-side modes. Because `h` is odd,
`h'` is even: every quantity in (A.11) is a secant average of `h'` on a
subinterval of `[-t,t]`. Consequently, if exact scalar bounds establish

\[
a<h'(s)<b\qquad(0\le s\le t),
\tag{A.12}
\]

then the same operator inequality holds on the entire rectangular tangent
space,

\[
aI\preceq A_Z\preceq bI.
\tag{A.13}
\]

This is the full-matrix lift. No diagonal certificate is being promoted into
a full-matrix claim.

The radial projector `P` in (A.9) lies in the diagonal singular-value block,
but (A.10) acts on every tangent block. Combining (A.10) with the
self-adjoint outer derivative gives

\[
\operatorname{Sym}DE_{h,\epsilon}(M)
=\frac{1-t}{\epsilon}
\left((1-t)A_Z+t\operatorname{Sym}(A_ZQ)\right).
\tag{A.14}
\]

## 4. Projection envelope

For any self-adjoint `A` satisfying `aI<=A<=bI` and any orthogonal projection
`Q`, the projection/anticommutator lemma gives

\[
\operatorname{Sym}(AQ)\succeq\Gamma(a,b)I.
\tag{A.15}
\]

When `a+b>0` and `b+3a>=0`, as in every band used below,

\[
\Gamma(a,b)=-\frac{(b-a)^2}{8(a+b)}.
\tag{A.16}
\]

For completeness, the general envelope used by the executable proof is

\[
\Gamma(a,b)=
\begin{cases}
-\dfrac{(b-a)^2}{8(a+b)},&a+b>0\text{ and }b+3a\ge0,\\[4pt]
\min\{a,0\},&\text{otherwise}.
\end{cases}
\tag{A.17}
\]

One proof evaluates the Rayleigh quotient on a unit vector `v`, sets
`beta=||Qv||`, and diagonalizes
`Sym((Qv) tensor v)`. Its only nonzero eigenvalues are
`beta(beta+1)/2` and `beta(beta-1)/2`. Applying the lower spectral bound `a`
on the positive eigenspace and the upper bound `b` on the negative eigenspace
leaves a scalar quadratic in `beta`; minimizing it on `[0,1]` yields
(A.17). This argument is dimension independent.

Substitution into (A.14) proves, whenever (A.12) holds,

\[
\operatorname{Sym}DE_{h,\epsilon}(M)
\succeq
\frac{1-t}{\epsilon}
\left((1-t)a+t\Gamma(a,b)\right)I.
\tag{A.18}
\]

## 5. Four-band interval certificate

The five-stage scalar value and derivative recurrences are

\[
x_{k+1}=q(x_k),\qquad
d_{k+1}=q'(x_k)d_k.
\tag{A.19}
\]

The primary certificate evaluates (A.19) with outward-rounded Arb balls,
without expanding the degree-3125 composition. A global adaptive dyadic cover
proves the strict upper bound

\[
h'(s)<b=\frac{4848763}{10000}
\qquad(0\le s\le1).
\tag{A.20}
\]

Three additional adaptive prefix covers and the inherited global lower cover
prove the following four rows. In each row, the scalar lower bound holds on
the entire prefix from zero through the band's right endpoint, so it covers
all secants in (A.11).

| normalized radius `t` | strict prefix lower `a` | `Gamma(a,b)` | exact band deficit upper |
| --- | ---: | ---: | ---: |
| `[0,1/200]` | `-504/5` | `-34301672838169/307261040000` | `504/5` |
| `[1/200,3/500]` | `-77877/500` | `-41040718127809/263297840000` | `1632191906745061351/10531913600000000` |
| `[3/500,63/10000]` | `-6379/40` | `-41518859781169/260321040000` | `491257553435828999/3099060000000000` |
| `[63/10000,1]` | `-199437/1250` | `-41528474059081/260261360000` | `6602082433275499863/41641817600000000` |

For a fixed row, negating the right side of (A.18) produces the exact
quadratic envelope

\[
\phi_{a}(t)=-(1-t)\bigl((1-t)a+t\Gamma(a,b)\bigr).
\tag{A.21}
\]

The generator checks both endpoints and the stationary point, when it lies in
the band, using exact rational arithmetic. The four maxima occur respectively
at `0`, `1/200`, `3/500`, and `63/10000`. The final row is active, giving

\[
\bar\delta_1=
\frac{6602082433275499863}{41641817600000000}
=158.544530805386839\ldots.
\tag{A.22}
\]

At both 160-bit and 224-bit precision, Arb accepts the same proof covers. The
global cover has 25,370 leaves and maximum dyadic depth 32; the prefix covers
have respectively 276, 367, and 909 leaves. A separate standard-library-only
reconstruction uses directed `Decimal` endpoint intervals, independently
rebuilds the scalar covers and all exact rational algebra, and reads the
canonical artifact only after its internal reconstruction is complete.

## 6. Promotion to a global pairwise certificate

The additive normalizer is globally `C1` by Section 2. Its derivative is
bounded by `1/epsilon`, and the polynomial spectral map has bounded derivative
on the closed Frobenius unit ball, so `E_(h,epsilon)` is globally Lipschitz.

For arbitrary `X,Y`, put `D=Y-X` and integrate along `X+theta D`. The origin
causes no exception because (A.8) is its ordinary Frechet derivative. Equations
(A.18)--(A.22) give

\[
\begin{aligned}
\langle E_{h,\epsilon}(Y)-E_{h,\epsilon}(X),D\rangle_F
&=\int_0^1
\langle DE_{h,\epsilon}(X+\theta D)D,D\rangle_F\,d\theta\\
&\ge-\frac{\bar\delta_1}{\epsilon}\lVert D\rVert_F^2.
\end{aligned}
\tag{A.23}
\]

This proves the full rectangular global upper bound (A.3), including line
segments through zero.

## 7. Exact rational swapped-diagonal witness

Define

\[
\tau=\frac{8974467}{1000000000},\qquad
u_-=\frac{803760}{1136689},\qquad
u_+=\frac{803761}{1136689}.
\tag{A.24}
\]

The direction is exactly Pythagorean:

\[
u_-^2+u_+^2=1.
\tag{A.25}
\]

At `epsilon=1`, set `r=tau/(1-tau)=8974467/991025533` and

\[
M_-=r\operatorname{diag}(u_-,u_+),\qquad
M_+=r\operatorname{diag}(u_+,u_-).
\tag{A.26}
\]

Both matrices have exact Frobenius norm `r`, so their additive denominators
are identical and their normalized radii are exactly `tau`. Direct rational
evaluation of all five polynomial stages gives the pairwise deficit

\[
-\frac{h(\tau u_+)-h(\tau u_-)}
{r(u_+-u_-)}
=158.1172496485127642979\ldots
>\frac{98823281}{625000}.
\tag{A.27}
\]

The reduced exact fraction has 49,624 numerator digits and 49,621 denominator
digits. Rather than embedding it in prose or JSON, the artifact records its
canonical fraction hash

```text
de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336
```

and separately verifies the small strict rational lower bound in (A.27).
Scaling both matrices in (A.26) by `epsilon` gives the lower witness in (A.4)
for every positive epsilon. Zero-padding embeds it in every rectangular shape
with at least two singular directions.

## 8. Repair and the deployed-epsilon obstruction

Equation (A.23) implies that

\[
M\longmapsto E_{h,\epsilon}(M)
+\frac{\bar\delta_1}{\epsilon}M
\tag{A.28}
\]

is globally monotone for every finite matrix shape. This conductance is only
certified sufficient. Conversely, the exact pair proves that every globally
valid constant repair in a shape with `min(m,n)>=2` must satisfy

\[
\rho\ge\delta_{m,n}(E_{h,\epsilon})
>\frac{98823281}{625000\epsilon}.
\tag{A.29}
\]

At the pinned upstream value `epsilon=10^-7`, the exact-real surrogate
therefore obeys

\[
\delta_{m,n}(E_{h,10^{-7}})>1,581,172,496
\qquad(\min\{m,n\}\ge2),
\tag{A.30}
\]

while the certified sufficient repair is

\[
\bar\rho_{10^{-7}}
=\frac{6602082433275499863}{4164181760}
=1,585,445,308.053868\ldots.
\tag{A.31}
\]

This is catastrophic as a global constant-repair design at ordinary raw
signal scale. For example, at `||M||_F=1`, the unrepaired spectral-polynomial
output is bounded in norm by `484.8763`, whereas the necessary linear repair
component already exceeds `1.581e9`. The conclusion is about the requested
global constant conductance; it is not a claim that every local trajectory or
every training run exhibits this worst case.

For comparison, the prior max-floor certificate at floor `c=1` needs only

\[
\frac{41528474059081}{260261360000}
=159.564501081070\ldots.
\tag{A.32}
\]

At `epsilon=1`, the four-band additive upper (A.22) is slightly smaller than
the max-floor upper. The obstruction is the exact `1/epsilon` scaling: at
`10^-7`, (A.31) is about `9.94` million times the max-floor `c=1` certificate.

## 9. Exact-real surrogate versus pinned BF16 Muon

The pinned upstream source is KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, with audited `muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
It casts the input to BF16, orients it, computes the BF16 Frobenius norm, adds
the Python-float `1e-7`, divides, and then runs five BF16 Jordan stages in the
locked operation order.

The theorem above certifies the continuous exact-real map (A.1), using the
same mathematical additive-denominator formula and exact rational polynomial
coefficients. It does **not** transfer to the literal upstream implementation:
BF16 casts make that backend map discontinuous, coefficients are represented
in BF16, and the recorded CPU witness even observes `1e-7` being absorbed by
denominator rounding. The upstream pin establishes formula provenance, not a
BF16 Jacobian, monotonicity, or repair theorem.

## 10. Boundary and negative controls

- At the origin, (A.8) is strictly positive; the proof does not create a
  spurious deficit from nondifferentiability of the norm.
- For the affine control `h(s)=s`, the tangential normalizer eigenvalue is
  `1/(r+epsilon)` and the radial eigenvalue is
  `epsilon/(r+epsilon)^2`, so additive normalization is monotone.
- As `r` tends to infinity, `t` tends to one and the outer factor `1-t` in
  (A.18) tends to zero. The closed endpoint `t=1` is a proof boundary, not a
  finite raw input.
- Every band maximum is checked against both exact endpoints and its possible
  stationary point. The active boundary is exactly `t=63/10000`.
- Adding any `rho` no larger than the strict endpoint in (A.29) leaves the
  locked pairwise gap negative. This is a failure witness for that repair,
  not merely a failure of the upper-bound method.

## Reproduce

```bash
uv run --locked python scripts/certify_additive_epsilon_deficit.py \
  --output results/summaries/additive_epsilon_deficit_certificate.json
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
```

The primary generator uses exact rational arithmetic and two independent Arb
precision passes. The reconstruction uses only the Python standard library
and directed endpoint rounding. Neither a sampled grid nor the BF16 witness
carries the global upper claim.

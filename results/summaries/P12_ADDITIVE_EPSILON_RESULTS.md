# P12 additive-epsilon deficit result

## Result

P12 certifies the exact-real additive-normalized five-step Jordan map

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right),
\qquad \epsilon>0,
\]

where

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad h=q^{\circ5}.
\]

For every fixed finite real matrix shape, the global pairwise deficit obeys
the exact scaling law

\[
\delta_{m,n}(E_{h,\epsilon})
=\frac{\delta_{m,n}(E_{h,1})}{\epsilon}.
\]

The four-band, full-rectangular certificate proves

\[
\delta_{m,n}(E_{h,\epsilon})
\le\frac1\epsilon
\frac{6602082433275499863}{41641817600000000}
=\frac{158.544530805386839\ldots}{\epsilon}
\]

for every finite positive `m,n`. This is a dimension-uniform global upper
certificate, not a sampled or diagonal estimate.

## Exact lower witness

The locked `2 x 2` pair uses

\[
\tau=\frac{8974467}{10^9},\qquad
u_-=\frac{803760}{1136689},\qquad
u_+=\frac{803761}{1136689},
\]

with `u_-^2+u_+^2=1`, raw radius
`r=tau/(1-tau)=8974467/991025533`, and swapped matrices

\[
M_-=r\operatorname{diag}(u_-,u_+),\qquad
M_+=r\operatorname{diag}(u_+,u_-).
\]

Exact five-stage rational evaluation proves

\[
-\frac{\langle E_{h,1}(M_+)-E_{h,1}(M_-),M_+-M_-\rangle_F}
{\lVert M_+-M_-\rVert_F^2}
=158.117249648512764\ldots
>\frac{98823281}{625000}=158.1172496.
\]

The canonical SHA-256 of the roughly 49,000-digit reduced exact quotient is

```text
de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336
```

The pair zero-pads into every shape with `min(m,n)>=2`. Accordingly, for all
such shapes,

\[
\frac{158.1172496}{\epsilon}
<\delta_{m,n}(E_{h,\epsilon})
\le\frac{158.544530805386839\ldots}{\epsilon}.
\]

The relative width from the strict lower endpoint is about `0.270231%`.
P12 does not claim an exact deficit or exact minimal repair for a fixed shape.

## Four-band upper certificate

The proof partitions normalized radius
`t=||M||_F/(epsilon+||M||_F)` into four bands. Outward-rounded Arb covers
bound every scalar derivative and therefore every active, divided-difference,
sum, repeated-singular-value, zero-singular-value, and rectangular null-side
mode of the spectral derivative.

| `t` band | strict prefix slope lower | exact deficit upper | maximizing `t` |
| --- | ---: | ---: | ---: |
| `[0,1/200]` | `-504/5` | `504/5` | `0` |
| `[1/200,3/500]` | `-77877/500` | `1632191906745061351/10531913600000000` | `1/200` |
| `[3/500,63/10000]` | `-6379/40` | `491257553435828999/3099060000000000` | `3/500` |
| `[63/10000,1]` | `-199437/1250` | `6602082433275499863/41641817600000000` | `63/10000` |

The last row is active. The scalar recurrences are enclosed independently at
160-bit and 224-bit Arb precision; both passes accept identical covers. The
global cover has 25,370 leaves, and the three prefix covers have 276, 367, and
909 leaves. Exact rational algebra then evaluates every projection envelope
and quadratic band maximum.

The origin is handled analytically, not log-sampled:

\[
DE_{h,\epsilon}(0)=\frac1\epsilon
\left(\frac{6889}{2000}\right)^5I\succ0.
\]

Global `C1` regularity permits direct line integration of the full symmetric
Jacobian bound into the authoritative pairwise inequality.

## Repair consequence

The certified sufficient constant repair is

\[
\rho_\epsilon=
\frac1\epsilon
\frac{6602082433275499863}{41641817600000000}.
\]

Then `E_(h,epsilon)(M)+rho_epsilon M` is globally monotone for every finite
matrix shape. The exact lower pair proves that any globally valid constant
repair in a shape containing it must be strictly greater than
`158.1172496/epsilon`.

At `epsilon=10^-7`, this becomes

| Quantity | Exact value | Decimal |
| --- | ---: | ---: |
| necessary strict lower endpoint | `1581172496` | `1,581,172,496` |
| certified sufficient repair | `6602082433275499863/4164181760` | `1,585,445,308.053868...` |

The two endpoints are close, and both are catastrophically large for a global
constant repair at ordinary raw signal scale. For any globally valid constant
repair in a shape with `min(m,n)>=2`, at unit Frobenius input the repair term
exceeds `1.581e9`; by contrast, the unrepaired polynomial output has the bound
`||H_h(N(M))||_F<484.8763`.

For comparison, the prior max-floor certificate at `c=1` has sufficient
upper

\[
\frac{41528474059081}{260261360000}
=159.564501081070\ldots.
\]

The additive map at `epsilon=1` is similarly repairable, with the slightly
smaller upper `158.544530805386...`. The deployed-scale obstruction is caused
by the exact `1/epsilon` law.

## Boundary and negative controls

- The exact origin derivative is positive.
- The affine control `h(s)=s` makes the additive normalizer monotone, with
  positive radial and tangential derivative eigenvalues.
- The raw infinite-radius limit sends `t` to one and the certified Jacobian
  envelope to zero.
- Every band endpoint and possible stationary point is checked exactly; the
  active boundary is `t=63/10000`.
- Every repair at or below the strict lower endpoint leaves the locked finite
  pair nonmonotone. This is a genuine pairwise negative control, not merely an
  infeasible certificate.

## Exact-real and BF16 scope

The formula is tied to pinned KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, audited `muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
That source casts to BF16 before its norm and five stages.

P12 certifies only the continuous exact-real surrogate with exact rational
coefficients. Literal upstream BF16 Muon is backend-specific and
discontinuous; its casts, rounded coefficients, epsilon absorption, current
normalization, and absence of this repair are outside the theorem. The
upstream pin establishes formula provenance, not transfer of the global
certificate.

## Evidence and limitations

The canonical machine-readable artifact is
`results/summaries/additive_epsilon_deficit_certificate.json`. Its primary
generator combines two outward-rounded Arb precision passes with exact
rational band and witness calculations. The independent reconstruction uses
only standard-library `Fraction` and directed `Decimal` intervals, completes
its reconstruction before reading the canonical artifact, and checks every
headline exact field.

```bash
uv run --locked python scripts/certify_additive_epsilon_deficit.py \
  --output results/summaries/additive_epsilon_deficit_certificate.json
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
```

No sampled grid supports the global upper claim. The older epsilon grid is a
lower-bound discovery artifact only.

The human proof audit is pending and unsigned. P12 does not establish a BF16
repair theorem, stability of literal upstream Muon, a useful repair at
`epsilon=10^-7`, momentum-loop convergence, stochastic training convergence,
or neural-network performance.

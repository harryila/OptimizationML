# P13 radial passivation tradeoff

## Status

P13 is a constructive exact-real continuation of P12. It replaces P12's
catastrophically large constant linear correction by an explicit radial
correction whose output grows only logarithmically with the raw signal norm.
The same result proves that no globally Lipschitz correction can remove the
inherited `1/epsilon` worst-case differential stiffness.

This is a theorem and certificate for a continuous real-arithmetic surrogate.
It is not a theorem for the discontinuous BF16 upstream implementation, and it
does not yet propagate the new radial correction through the P7--P11 stability
stack. The human proof audit remains pending and unsigned.

## Locked operator and inherited certificate

For every fixed finite real matrix shape and `epsilon>0`, P13 retains P12's
operator

\[
E_{h,\epsilon}(M)=\mathcal H_h\!\left(
  \frac{M}{\lVert M\rVert_F+\epsilon}
\right),
\]

where `h` is exactly five compositions of the Jordan quintic with rational
coefficients

\[
\left(\frac{6889}{2000},-\frac{191}{40},\frac{4063}{2000}\right).
\]

The construction imports the following exact constants from P12:

\[
\begin{aligned}
\bar\delta_1
  &=\frac{6602082433275499863}{41641817600000000},\\
a&=-\frac{199437}{1250},\\
\Gamma&=-\frac{41528474059081}{260261360000},\\
z_0&=\frac{63}{9937}.
\end{aligned}
\]

Here `bar_delta_1=158.544530805386839...` is P12's dimension-uniform
unit-epsilon upper certificate, and `a,Gamma` are the scalar and projection
lower bounds in its last radius band. The relation

\[
\frac{z_0}{1+z_0}=\frac{63}{10000}
\]

maps the P13 breakpoint exactly to that band's left endpoint.

## Radial correction

Set `z=||M||_F/epsilon` and define

\[
\widehat d(z)=
\begin{cases}
\bar\delta_1, & 0\le z\le z_0,\\[3pt]
-\dfrac{a+\Gamma z}{(1+z)^2}, & z\ge z_0.
\end{cases}
\]

The two branches agree exactly at `z_0`. The tail is positive and strictly
decreasing because

\[
\widehat d'(z)=
\frac{2a-\Gamma+\Gamma z}{(1+z)^3}<0
\qquad (z\ge z_0).
\]

This function majorizes the complete pointwise full-matrix deficit envelope
proved in P12. Define

\[
p(z)=\int_0^z\widehat d(u)\,du
\]

and, with `G_epsilon(0)=0`,

\[
G_\epsilon(M)=p\!\left(\frac{\lVert M\rVert_F}{\epsilon}\right)
\frac{M}{\lVert M\rVert_F}.
\]

The primitive is linear below the breakpoint. Above it, the closed form is

\[
\begin{aligned}
p(z)={}&\bar\delta_1z_0
-\Gamma\log\!\left(\frac{1+z}{1+z_0}\right)\\
&+(a-\Gamma)
\left(\frac{1}{1+z}-\frac{1}{1+z_0}\right).
\end{aligned}
\]

For nonzero `M`, the radial and tangential derivative eigenvalues are

\[
\lambda_{\mathrm{rad}}=
\frac{\widehat d(z)}{\epsilon},
\qquad
\lambda_{\mathrm{tan}}=
\frac{p(z)}{\epsilon z}.
\]

Since `d_hat` is positive and nonincreasing,

\[
0<\widehat d(z)\le \frac{p(z)}{z}\le\bar\delta_1.
\]

The origin is not singular: near zero,
`G_epsilon(M)=(bar_delta_1/epsilon)M`. The two formulas also match in first
derivative at `z_0`. Consequently `G_epsilon` is globally continuously
differentiable and

\[
\operatorname{Lip}(G_\epsilon)
=\frac{\bar\delta_1}{\epsilon}.
\]

Combining its smallest derivative eigenvalue with P12's pointwise symmetric-
Jacobian lower bound gives

\[
\operatorname{Sym}D(E_{h,\epsilon}+G_\epsilon)(M)\succeq0.
\]

Integration on line segments therefore proves the authoritative pairwise
claim:

\[
E_{h,\epsilon}+G_\epsilon
\quad\text{is globally monotone}
\]

for every finite real rectangular matrix shape.

## Output-magnitude improvement

The primary certificate evaluates the logarithmic primitive with outward-
rounded Arb arithmetic at two working precisions. The independent
standard-library reconstruction instead uses exact rational atanh-series
enclosures for every logarithm.

At the pinned `epsilon=1e-7`, the certified values are summarized below.

| Raw Frobenius norm | Radial correction norm | Constant P12 correction norm | Reduction factor |
| ---: | ---: | ---: | ---: |
| `1` | `2571.857826470212...` | `1,585,445,308.053868...` | `616,459.16...` |
| `13872266672489/549755813888 = 25.2335060804...` | `3086.959580254301...` | `40,006,343,820.950377...` | `12,959,788.68...` |

The second row uses P11's exact `V<=1` signal guard, rather than rounding that
guard to `25.2335`. Thus the nonlinear correction removes the catastrophic
linear growth in output magnitude over the recorded operating scale. It does
not make the correction uniformly bounded: as `z` tends to infinity,
`d_hat(z)` tends to zero while

\[
p(z)=-\Gamma\log z+O(1).
\]

Hence the correction norm grows logarithmically and its gain relative to the
raw input tends to zero.

## Unavoidable stiffness

Let `C_epsilon` be any globally Lipschitz correction for which
`E_{h,epsilon}+C_epsilon` is globally monotone in a shape containing P12's
locked rank-two witness. Applying monotonicity to that pair and then
Cauchy--Schwarz gives

\[
\operatorname{Lip}(C_\epsilon)
\ge \delta_{\mathrm{pair}}(E_{h,\epsilon})
>\frac{98823281}{625000\,\epsilon}
=\frac{158.1172496}{\epsilon}.
\]

This universal lower bound zero-pads only into shapes with
`min(m,n)>=2`; it is not asserted for scalar-only shapes. The constructed
radial repair has the matching `1/epsilon` order, with exact Lipschitz constant
`bar_delta_1/epsilon`, only `0.270231%` above the strict witness lower.
Nonlinearity therefore improves output magnitude but cannot eliminate
worst-case differential stiffness.

## Explicit-step negative control

The stiffness is visible in the scalar curvature-one quadratic under the
pinned exact-real EMA/Nesterov ordering

\[
\begin{aligned}
m_+&=\beta m+(1-\beta)w,\\
s_+&=\beta m_++(1-\beta)w,\\
w_+&=w-\eta R_\epsilon(s_+),
\end{aligned}
\qquad \beta=\frac{19}{20}.
\]

For an operator with origin slope `k/epsilon`, its exact Jury boundary is

\[
\eta<\frac{2(1+\beta)\epsilon}
{(1-\beta)(1+2\beta)k}.
\]

For the radial correction alone, `k=bar_delta_1`, and the exact threshold at
`epsilon=1e-7` is

\[
\frac{1082687257600}{63820130188329832009}
=1.696466701658312\ldots\times10^{-8}.
\]

For the full repaired operator,
`k=(6889/2000)^5+bar_delta_1`, and the threshold becomes approximately
`4.180242693e-9`. The prior `eta=1/32000` violates both boundaries by exact
negative Jury margins. A finite rational control with
`m0=0,w0=1/389025000` has `s1=epsilon/399` inside the repair's linear core and
expands the scalar objective by a factor strictly above `23,325,554` in one
step. These are negative controls for explicit treatment of the stiff radial
repair, not claims about every initialization or an implicit/resolvent
implementation.

## Scope and verification

The result covers arbitrary finite matrix shapes in exact real arithmetic.
The P12 rank-two lower witness retains its `min(m,n)>=2` qualification. No
sampling establishes the global result; numerical matrix checks, if run, are
falsification diagnostics only.

The pinned upstream revision is used solely to identify the practical
additive-epsilon formula and its cast/order provenance. Upstream casts to BF16
before the norm and polynomial stages, and its resulting map is backend-
specific and discontinuous. Neither P12's Jacobian certificate nor P13's
radial passivation theorem transfers to that implementation without a
separate finite-precision analysis.

The canonical artifact records exact identities, two-precision Arb magnitude
enclosures, the P12 pair lower bound, the scalar negative control, prior-
artifact hashes, source hashes, software, hardware, Git state, and the absence
of randomness. Reproduce and independently check it with

```bash
uv run --locked python scripts/certify_radial_passivation_tradeoff.py \
  --output results/summaries/radial_passivation_tradeoff_certificate.json
uv run --locked python scripts/reconstruct_radial_passivation_tradeoff.py \
  --require-canonical
```

The reconstruction imports only the Python standard library, rebuilds the
exact rational algebra before reading the canonical JSON, and encloses the
logarithms independently. It is an independent automated reconstruction, not
a substitute for the pending human proof audit.

import re

import ipdb

_RAW = r"""
\alpha α
\beta β
\Gamma Γ
\gamma γ  
\Delta Δ
\delta δ
\epsilon ∊
\zeta ζ
\eta η
\Theta Θ
\theta θ
\iota ι
\kappa κ
\Lambda Λ
\lambda λ
\mu μ
\nu ν
\Xi Ξ
\xi ξ
\Pi Π
\pi π
\rho ρ
\Sigma Σ
\sigma σ
\tau τ
\Upsilon Υ
\upsilon υ
\Phi Φ
\phi ϕ
\chi χ
\Psi Ψ
\psi ψ
\Omega Ω
\omega ω
\vartheta ϑ
\varsigma ς
\varrho ϱ
\varpropto ∝
\varpi ϖ
\varphi φ
\varnothing ∅
\varkappa ϰ
\varepsilon ε
\sqrt √
"""


def _parse_raw(raw: str) -> dict[str, str]:
    out = {}
    for line in raw.strip().split("\n"):
        syms = line.split(" ")
        out[syms[0]] = syms[1]
    return out


_CVT = _parse_raw(_RAW)


def from_mathtext(x: str) -> str:
    """Replace mathtext with unicode."""
    # 1: Identify text between matching dollar signs.
    dollar_idxs = [m.start() for m in re.finditer(r"\$", x)]
    if len(dollar_idxs) % 2 != 0:
        raise ValueError(f"Expected matching $, but found odd number!  {x}")

    out = []
    prev_e_idx = -1
    for s_idx, e_idx in zip(dollar_idxs[::2], dollar_idxs[1::2]):
        # Add the text between prev_e_idx and s_idx.
        out.append(x[prev_e_idx + 1 : s_idx])

        substr = x[s_idx + 1 : e_idx]
        for mathtext, unicode in _CVT.items():
            substr = substr.replace(mathtext, unicode)
        out.append(substr)

        prev_e_idx = e_idx

    # Add the text from the last e_idx to the end.
    out.append(x[prev_e_idx + 1 :])
    joined = "".join(out)

    # Replace curly braces.
    joined = joined.replace("{", "").replace("}", "")

    return joined

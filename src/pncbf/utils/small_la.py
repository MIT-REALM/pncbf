import copy
from dataclasses import dataclass
from typing import Callable, List

import jax
import jax.numpy as jnp


def inv22(mat):
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return jnp.array([[m4, -m2], [-m3, m1]]) * inv_det


def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    inv_det = 1.0 / det
    return (
        jnp.array(
            [
                [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
            ]
        )
        * inv_det
    )


def inv44(m):
    """
    See https://github.com/willnode/N-Matrix-Programmer
    for source in the "Info" folder
    MIT License.
    """
    A2323 = m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]
    A1323 = m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]
    A1223 = m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]
    A0323 = m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]
    A0223 = m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]
    A0123 = m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]
    A2313 = m[1, 2] * m[3, 3] - m[1, 3] * m[3, 2]
    A1313 = m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]
    A1213 = m[1, 1] * m[3, 2] - m[1, 2] * m[3, 1]
    A2312 = m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]
    A1312 = m[1, 1] * m[2, 3] - m[1, 3] * m[2, 1]
    A1212 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    A0313 = m[1, 0] * m[3, 3] - m[1, 3] * m[3, 0]
    A0213 = m[1, 0] * m[3, 2] - m[1, 2] * m[3, 0]
    A0312 = m[1, 0] * m[2, 3] - m[1, 3] * m[2, 0]
    A0212 = m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]
    A0113 = m[1, 0] * m[3, 1] - m[1, 1] * m[3, 0]
    A0112 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]

    det = (
        m[0, 0] * (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223)
        - m[0, 1] * (m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223)
        + m[0, 2] * (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123)
        - m[0, 3] * (m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123)
    )
    invdet = 1.0 / det

    return invdet * jnp.array(
        [
            (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223),
            -(m[0, 1] * A2323 - m[0, 2] * A1323 + m[0, 3] * A1223),
            (m[0, 1] * A2313 - m[0, 2] * A1313 + m[0, 3] * A1213),
            -(m[0, 1] * A2312 - m[0, 2] * A1312 + m[0, 3] * A1212),
            -(m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223),
            (m[0, 0] * A2323 - m[0, 2] * A0323 + m[0, 3] * A0223),
            -(m[0, 0] * A2313 - m[0, 2] * A0313 + m[0, 3] * A0213),
            (m[0, 0] * A2312 - m[0, 2] * A0312 + m[0, 3] * A0212),
            (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123),
            -(m[0, 0] * A1323 - m[0, 1] * A0323 + m[0, 3] * A0123),
            (m[0, 0] * A1313 - m[0, 1] * A0313 + m[0, 3] * A0113),
            -(m[0, 0] * A1312 - m[0, 1] * A0312 + m[0, 3] * A0112),
            -(m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123),
            (m[0, 0] * A1223 - m[0, 1] * A0223 + m[0, 2] * A0123),
            -(m[0, 0] * A1213 - m[0, 1] * A0213 + m[0, 2] * A0113),
            (m[0, 0] * A1212 - m[0, 1] * A0212 + m[0, 2] * A0112),
        ]
    ).reshape((4, 4))


def fast_dot(a, b):
    return (a * b).sum()


fast_mat_mul = jax.vmap(jax.vmap(fast_dot, in_axes=(None, 1)), in_axes=(0, None))


def inv_recurse(mat):
    if mat.shape[0] == 1:
        return 1.0 / mat
    if mat.shape[0] == 2:
        return inv22(mat)
    elif mat.shape[0] == 3:
        return inv33(mat)
    elif mat.shape[0] == 4:
        return inv44(mat)
    r = 4
    A = mat[:r, :r]
    B = mat[:r, r:]
    C = mat[r:, :r]
    D = mat[r:, r:]
    A_inv = inv_recurse(A)
    CA_inv = fast_mat_mul(C, A_inv)
    schur = D - fast_mat_mul(CA_inv, B)
    schur_inv = inv_recurse(schur)
    A_invB = fast_mat_mul(A_inv, B)
    lr = schur_inv
    ur = -fast_mat_mul(A_invB, schur_inv)
    ll = -fast_mat_mul(schur_inv, CA_inv)
    ul = A_inv - fast_mat_mul(A_invB, ll)
    return jnp.concatenate(
        (
            jnp.concatenate((ul, ur), axis=1),
            jnp.concatenate((ll, lr), axis=1),
        ),
        axis=0,
    )


def get_inv_fnc(d):
    inv = {1: lambda m: 1.0 / m, 2: inv22, 3: inv33, 4: inv44}
    return inv.get(d, inv_recurse)


class CodeGenCtx:
    def __init__(self):
        self.assignments = []
        self.definitions = dict()

    def assign(self, name, definition):
        self.assignments.append(name)
        self.definitions[name] = definition
        return self.assignments[-1]

    def lines(self):
        return [f"{a} = {self.definitions[a]}" for a in self.assignments]


@dataclass
class Spec:
    prefix: str
    generator: Callable[[CodeGenCtx, list, int], str]
    in_types: List[str]
    out_types: List[str]


def gen_lu(ctx, args, d):
    U = copy.deepcopy(args[0])
    L = [[None] * d for i in range(d)]
    for k in range(d - 1):
        inv_k = ctx.assign(f"inv_{k}", f"1.0 / {U[k][k]}")
        for j in range(k + 1, d):
            L[j][k] = ctx.assign(f"L_{j}{k}", f"{U[j][k]} * {inv_k}")
        for i in range(k + 1, d):
            for j in range(k + 1, d):
                if i == k + 1:
                    name = f"U_{i}{j}"
                else:
                    name = f"U_{k}_{i}{j}"
                U[i][j] = ctx.assign(name, f"{U[i][j]} - {U[k][j]} * {U[i][k]} * {inv_k}")
    LU = [[U[i][j] if i <= j else L[i][j] for j in range(d)] for i in range(d)]
    return [LU]


lu_spec = Spec("lu", gen_lu, in_types=("M",), out_types=("M",))


def transpose(A):
    d = len(A)
    return [[A[j][i] for j in range(d)] for i in range(d)]


def gen_lu_inv(ctx, args, d):
    invU = copy.deepcopy(args[0])
    for k in range(d)[::-1]:
        invU[k][k] = ctx.assign(f"invU_{k}{k}", f"1.0 / {invU[k][k]}")
        for j in range(k + 1, d):
            invU[k][j] = ctx.assign(f"invU_{k}{j}", f"{invU[k][j]} * {invU[k][k]}")
        for i in range(k):
            mult = f"-{invU[i][k]}"
            invU[i][k] = ctx.assign(f"invU_{k}_{i}{k}", f"{mult} * {invU[k][k]}")
            for j in range(k + 1, d):
                invU[i][j] = ctx.assign(f"invU_{k}_{i}{j}", f"{invU[i][j]} + {mult} * {invU[k][j]}")

    invLU_T = transpose(invU)
    for i in range(d - 1):
        for j in range(i + 1, d):
            invLU_T[i][j] = "0"
    for k in range(d)[::-1]:
        for i in range(k):
            mult = f"-{invLU_T[k][i]}"
            for j in range(d):
                name = f"invLU_T_{k}_{i}{j}"
                invLU_T[i][j] = ctx.assign(name, f"{invLU_T[i][j]} + {mult} * {invLU_T[k][j]}")
    return [transpose(invLU_T)]


lu_inv_spec = Spec("lu_inv", gen_lu_inv, in_types=("M",), out_types=("M",))


def gen_lu_solve(ctx, args, d):
    LU, B = args
    Y = [None] * d
    for i in range(d):
        terms_i = [f"-{LU[i][j]}*{Y[j]}" for j in range(i)]
        Y[i] = ctx.assign(f"Y_{i}", f"{B[i]}" + "".join(terms_i))
    X = [None] * d
    for i in range(d)[::-1]:
        invkk = ctx.assign(f"inv_{i}", f"1.0 / {LU[i][i]}")
        terms_i = [f"-{LU[i][j]}*{X[j]}*{invkk}" for j in range(i + 1, d)]
        X[i] = ctx.assign(f"X_{i}", f"{Y[i]}*{invkk}" + "".join(terms_i))
    return [X]


lu_solve_spec = Spec("lu_solve", gen_lu_solve, in_types=("M", "v"), out_types=("v"))


def gen_solve(ctx, args, d):
    LU = gen_lu(ctx, args[:1], d)[0]
    return gen_lu_solve(ctx, [LU, args[1]], d)


solve_spec = Spec("solve", gen_solve, in_types=("M", "v"), out_types=("v"))


def build_linalg(spec, d, print_code=True):
    assert d < 10
    ctx = CodeGenCtx()
    args = []
    arg_spec = []
    for k, entry in enumerate(spec.in_types):
        if entry == "M":
            args.append([[f"m{k}[{i}, {j}]" for j in range(d)] for i in range(d)])
            arg_spec.append(f"m{k}")
        elif entry == "v":
            args.append([f"v{k}[{i}]" for i in range(d)])
            arg_spec.append(f"v{k}")
    out_names = spec.generator(ctx, args, d)
    lines = ctx.lines()
    ret_vals = []
    for k, entry in enumerate(spec.out_types):
        if entry == "M":
            ret_vals.append(
                "jnp.array([" + ", ".join(["[" + ", ".join(out_names[k][i]) + "]" for i in range(d)]) + "])"
            )
        elif entry == "v":
            ret_vals.append("jnp.array([" + ", ".join(out_names[k]) + "])")
        elif entry == "s":
            ret_vals.append(f"{out_names[k]}")
    if len(ret_vals) == 1:
        lines.append(f"return {ret_vals[0]}")
    else:
        lines.append("return (" + "".join([v + "," for v in ret_vals]) + ")")

    name = f"{spec.prefix}{d}"
    arg_str = ",".join(arg_spec)
    lines = [f"def {name}({arg_str}):"] + ["    " + L for L in lines]
    code = "\n".join(lines)
    if print_code:
        print(code)
    return name, code


def gen(name):
    if name in globals():
        return globals()[name]
    for spec in [lu_spec, lu_spec, lu_inv_spec, lu_solve_spec, solve_spec]:
        if not name.startswith(spec.prefix):
            continue
        d = int(name[len(spec.prefix) :])
        name, code = build_linalg(spec, d, print_code=False)
        exec(code)
        globals()[name] = locals()[name]
        return globals()[name]


@jax.jit
def logdet(A):
    d = A.shape[0]
    lu = gen(f"lu{d}")(A)
    return jnp.sum(jnp.log(jnp.abs(jnp.diag(lu))))

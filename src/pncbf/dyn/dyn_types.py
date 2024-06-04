from typing import TypeVar

from jaxtyping import Bool

from pncbf.utils.jax_types import Arr, Float

State = Float[Arr, "nx"]
EncState = Float[Arr, "enc_nx"]
Control = Float[Arr, "nu"]
Disturb = Float[Arr, "nd"]
Obs = Float[Arr, "nobs"]
PolObs = Float[Arr, "npolobs"]
VObs = Float[Arr, "nVobs"]
Sample = Float[Arr, "*"]
Param = TypeVar("Param")

BState = Float[Arr, "b nx"]
BControl = Float[Arr, "b nu"]
BDisturb = Float[Arr, "b nd"]
BObs = Float[Arr, "b nobs"]
BPolObs = Float[Arr, "b npolobs"]
BVObs = Float[Arr, "b nVobs"]
BParam = TypeVar("BParam")

SState = Float[Arr, "s nx"]
SControl = Float[Arr, "s nx"]
SControl = Float[Arr, "s nx"]

STState = Float[Arr, "s T nx"]

LFloat = Float[Arr, "nl"]
TLFloat = Float[Arr, "T nl"]
BLFloat = Float[Arr, "b nl"]
BTLFloat = Float[Arr, "b T nl"]
ZBLFloat = Float[Arr, "nz b nl"]


HFloat = Float[Arr, "nh"]
BHFloat = Float[Arr, "b nh"]
BTHFloat = Float[Arr, "b T nh"]
ZBHFloat = Float[Arr, "nz b nh"]

HState = Float[Arr, "nh nx"]

BBState = Float[Arr, "b1 b2 nx"]
BBControl = Float[Arr, "b1 b2 nu"]
BBDisturb = Float[Arr, "b1 b2 nd"]
BBParam = TypeVar("BBParam")

BBLFloat = Float[Arr, "b1 b2 nl"]

BBTControl = Float[Arr, "b1 b2 T nu"]

ZFloat = Float[Arr, "nz"]
ZBFloat = Float[Arr, "nz b"]
ZBBFloat = Float[Arr, "nz b1 b2"]
ZBTState = Float[Arr, "nz b T nx"]
ZBBControl = Float[Arr, "nz b1 b2 nu"]

THFloat = Float[Arr, "T nh"]
ZBTHFloat = Float[Arr, "nz b T nh"]

BBHFloat = Float[Arr, "b1 b2 nh"]
ZBBHFloat = Float[Arr, "nz b1 b2 nh"]
BBTHFloat = Float[Arr, "b1 b2 T nh"]

BBTState = Float[Arr, "b1 b2 T nx"]

BBHBool = Bool[Arr, "b1 b2 nh"]

BTState = Float[Arr, "b T nx"]
BTControl = Float[Arr, "b T nu"]
BTObs = Float[Arr, "b T nobs"]
BTVObs = Float[Arr, "b T nVobs"]
BTPolObs = Float[Arr, "b T npolobs"]
BTSample = Float[Arr, "b T *"]

BTEFloat = Float[Arr, "b T e"]
BTEState = Float[Arr, "b T e nx"]

FxShape = Float[Arr, "nx nx"]
FuShape = Float[Arr, "nx nu"]

TState = Float[Arr, "T nx"]
TControl = Float[Arr, "T nu"]

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import torch
import torchsde
from torchsde.settings import LEVY_AREA_APPROXIMATIONS

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_complex_brownian_dtype_mapping():
    size = (8, 3)
    bm32 = torchsde.ComplexBrownian(t0=0.0, t1=1.0, size=size, dtype=torch.float32, device=device, entropy=11)
    bm64 = torchsde.ComplexBrownian(t0=0.0, t1=1.0, size=size, dtype=torch.float64, device=device, entropy=12)
    bmc64 = torchsde.ComplexBrownian(t0=0.0, t1=1.0, size=size, dtype=torch.complex64, device=device, entropy=13)
    bmc128 = torchsde.ComplexBrownian(t0=0.0, t1=1.0, size=size, dtype=torch.complex128, device=device, entropy=14)

    assert bm32.dtype == torch.complex64
    assert bm64.dtype == torch.complex128
    assert bmc64.dtype == torch.complex64
    assert bmc128.dtype == torch.complex128


def test_complex_brownian_reproducible_with_entropy():
    kwargs = dict(t0=0.0, t1=1.0, size=(16, 4), dtype=torch.float64, device=device, entropy=12345)
    bm1 = torchsde.ComplexBrownian(**kwargs)
    bm2 = torchsde.ComplexBrownian(**kwargs)

    dW1 = bm1(0.1, 0.8)
    dW2 = bm2(0.1, 0.8)
    torch.testing.assert_close(dW1, dW2, rtol=0.0, atol=0.0)


def test_complex_brownian_tuple_outputs_are_complex_and_shaped():
    size = (10, 3)
    bm = torchsde.ComplexBrownian(
        t0=0.0,
        t1=1.0,
        size=size,
        dtype=torch.float64,
        device=device,
        entropy=77,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.foster,
    )
    dW, U, A = bm(0.2, 0.9, return_U=True, return_A=True)

    assert dW.shape == size
    assert U.shape == size
    assert A.shape == (*size, size[-1])
    assert dW.is_complex()
    assert U.is_complex()
    assert A.is_complex()


def test_complex_brownian_real_imag_are_non_degenerate_and_not_identical():
    bm = torchsde.ComplexBrownian(t0=0.0, t1=1.0, size=(8192, 1), dtype=torch.float64, device=device, entropy=999)
    dW = bm(0.0, 1.0)
    real = dW.real.reshape(-1)
    imag = dW.imag.reshape(-1)

    assert real.var(unbiased=False) > 0
    assert imag.var(unbiased=False) > 0
    assert not torch.allclose(real, imag)

    real_centered = real - real.mean()
    imag_centered = imag - imag.mean()
    corr = (real_centered * imag_centered).mean() / (
        real_centered.square().mean().sqrt() * imag_centered.square().mean().sqrt()
    )
    assert corr.abs().item() < 0.1

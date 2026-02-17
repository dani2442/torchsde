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

torch.manual_seed(1234)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ComplexOULearnable(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(
            torch.tensor([0.5 + 0.3j, 1.0 + 0.0j], dtype=torch.complex128, device=device)
        )
        self.sigma = torch.nn.Parameter(
            torch.tensor([0.5 + 0.2j, 0.3 + 0.0j], dtype=torch.complex128, device=device)
        )

    def f(self, t, y):
        return -self.alpha * y

    def g(self, t, y):
        return self.sigma.unsqueeze(0).expand_as(y)


class _ComplexOURealParams(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self):
        super().__init__()
        self.alpha_real = torch.nn.Parameter(torch.tensor([0.5, 1.0], device=device))
        self.alpha_imag = torch.nn.Parameter(torch.tensor([0.3, 0.0], device=device))
        self.sigma_real = torch.nn.Parameter(torch.tensor([0.5, 0.3], device=device))
        self.sigma_imag = torch.nn.Parameter(torch.tensor([0.2, 0.0], device=device))

    def _alpha(self):
        return torch.complex(self.alpha_real, self.alpha_imag)

    def _sigma(self):
        return torch.complex(self.sigma_real, self.sigma_imag)

    def f(self, t, y):
        return -self._alpha() * y

    def g(self, t, y):
        return self._sigma().unsqueeze(0).expand_as(y)


class _ComplexOUStratLearnable(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(
            torch.tensor([0.5 + 0.3j, 1.0 + 0.0j], dtype=torch.complex128, device=device)
        )
        self.sigma = torch.nn.Parameter(
            torch.tensor([0.5 + 0.2j, 0.3 + 0.0j], dtype=torch.complex128, device=device)
        )

    def f(self, t, y):
        return -self.alpha * y

    def g(self, t, y):
        return self.sigma.unsqueeze(0).expand_as(y)


class _ComplexGeneralSDE(torch.nn.Module):
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, d, m):
        super().__init__()
        g_real = torch.randn(d, m, device=device) * 0.3
        g_imag = torch.randn(d, m, device=device) * 0.3
        self.register_buffer("g_mat", torch.complex(g_real, g_imag))

    def f(self, t, y):
        return -0.5 * y

    def g(self, t, y):
        return self.g_mat.unsqueeze(0).expand(y.shape[0], -1, -1)


def _setup_problem():
    d = 2
    batch_size = 64
    t0, t1 = 0.0, 0.5
    n_steps = 200
    dt = (t1 - t0) / n_steps
    ts = torch.linspace(t0, t1, n_steps + 1, dtype=torch.float64, device=device)
    y0 = torch.full((batch_size, d), 1.0 + 0.5j, dtype=torch.complex128, device=device)
    return d, batch_size, dt, ts, y0


def test_complex_adjoint_matches_backprop_complex_params():
    _, batch_size, dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOULearnable()
    bm_bp = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=42
    )
    ys_bp = torchsde.sdeint(sde_bp, y0, ts, dt=dt, method="euler", bm=bm_bp)
    loss_bp = ys_bp[-1].abs().pow(2).sum()
    loss_bp.backward()
    grad_alpha_bp = sde_bp.alpha.grad.detach().clone()
    grad_sigma_bp = sde_bp.sigma.grad.detach().clone()

    sde_adj = _ComplexOULearnable()
    with torch.no_grad():
        sde_adj.alpha.copy_(sde_bp.alpha.detach())
        sde_adj.sigma.copy_(sde_bp.sigma.detach())
    bm_adj = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=42
    )
    ys_adj = torchsde.sdeint_adjoint(
        sde_adj, y0, ts, dt=dt, method="euler", adjoint_method="euler", bm=bm_adj
    )
    loss_adj = ys_adj[-1].abs().pow(2).sum()
    loss_adj.backward()
    grad_alpha_adj = sde_adj.alpha.grad.detach().clone()
    grad_sigma_adj = sde_adj.sigma.grad.detach().clone()

    torch.testing.assert_close(ys_bp, ys_adj, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(loss_bp, loss_adj, rtol=1e-10, atol=1e-10)
    assert grad_alpha_adj.is_complex()
    assert grad_sigma_adj.is_complex()
    assert torch.isfinite(torch.view_as_real(grad_alpha_adj)).all()
    assert torch.isfinite(torch.view_as_real(grad_sigma_adj)).all()
    assert grad_alpha_adj.abs().sum() > 0
    assert grad_sigma_adj.abs().sum() > 0

    alpha_rel = (grad_alpha_bp - grad_alpha_adj).abs().max() / grad_alpha_bp.abs().max()
    sigma_rel = (grad_sigma_bp - grad_sigma_adj).abs().max() / grad_sigma_bp.abs().max()
    assert alpha_rel.item() < 0.03
    assert sigma_rel.item() < 0.03


def test_complex_adjoint_matches_backprop_real_params():
    _, batch_size, dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOURealParams()
    bm_bp = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=99
    )
    ys_bp = torchsde.sdeint(sde_bp, y0, ts, dt=dt, method="euler", bm=bm_bp)
    loss_bp = ys_bp[-1].abs().pow(2).sum()
    loss_bp.backward()
    grads_bp = {n: p.grad.detach().clone() for n, p in sde_bp.named_parameters()}

    sde_adj = _ComplexOURealParams()
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(sde_bp.named_parameters(), sde_adj.named_parameters()):
            assert n1 == n2
            p2.copy_(p1.detach())
    bm_adj = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=99
    )
    ys_adj = torchsde.sdeint_adjoint(
        sde_adj, y0, ts, dt=dt, method="euler", adjoint_method="euler", bm=bm_adj
    )
    loss_adj = ys_adj[-1].abs().pow(2).sum()
    loss_adj.backward()
    grads_adj = {n: p.grad.detach().clone() for n, p in sde_adj.named_parameters()}

    torch.testing.assert_close(ys_bp, ys_adj, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(loss_bp, loss_adj, rtol=1e-10, atol=1e-10)

    for name in grads_bp:
        g_bp = grads_bp[name]
        g_adj = grads_adj[name]
        assert g_adj.dtype == g_bp.dtype
        assert not g_adj.is_complex()
        assert torch.isfinite(g_adj).all()
        rel = (g_bp - g_adj).abs().max() / g_bp.abs().max().clamp(min=1e-8)
        assert rel.item() < 0.05


def test_complex_adjoint_stratonovich_midpoint():
    _, batch_size, dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOUStratLearnable()
    bm_bp = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=77
    )
    ys_bp = torchsde.sdeint(sde_bp, y0, ts, dt=dt, method="midpoint", bm=bm_bp)
    loss_bp = ys_bp[-1].abs().pow(2).sum()
    loss_bp.backward()
    grad_alpha_bp = sde_bp.alpha.grad.detach().clone()

    sde_adj = _ComplexOUStratLearnable()
    with torch.no_grad():
        sde_adj.alpha.copy_(sde_bp.alpha.detach())
        sde_adj.sigma.copy_(sde_bp.sigma.detach())
    bm_adj = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(batch_size, 2), dtype=torch.float64, device=device, entropy=77
    )
    ys_adj = torchsde.sdeint_adjoint(
        sde_adj, y0, ts, dt=dt, method="midpoint", adjoint_method="midpoint", bm=bm_adj
    )
    loss_adj = ys_adj[-1].abs().pow(2).sum()
    loss_adj.backward()
    grad_alpha_adj = sde_adj.alpha.grad.detach().clone()

    torch.testing.assert_close(ys_bp, ys_adj, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(loss_bp, loss_adj, rtol=1e-10, atol=1e-10)
    rel = (grad_alpha_bp - grad_alpha_adj).abs().max() / grad_alpha_bp.abs().max()
    assert rel.item() < 0.03


def test_complex_general_noise_output_dtype():
    batch_size = 32
    d = 4
    m = 3
    t0, t1 = 0.0, 0.5
    n_steps = 50
    dt = (t1 - t0) / n_steps

    ts = torch.linspace(t0, t1, n_steps + 1, dtype=torch.float64, device=device)
    y0 = torch.full((batch_size, d), 1.0 + 0.5j, dtype=torch.complex128, device=device)
    sde = _ComplexGeneralSDE(d=d, m=m)

    ys = torchsde.sdeint(sde, y0, ts, dt=dt, method="euler")
    assert ys.shape == (n_steps + 1, batch_size, d)
    assert ys.is_complex()

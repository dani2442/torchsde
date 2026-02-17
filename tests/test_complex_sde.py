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
    return dt, ts, y0


def _copy_parameters(src, dst):
    with torch.no_grad():
        for (src_name, src_param), (dst_name, dst_param) in zip(src.named_parameters(), dst.named_parameters()):
            assert src_name == dst_name
            dst_param.copy_(src_param.detach())


def _run_solver(sde, y0, ts, dt, method, entropy, use_adjoint=False, noise_size=None, adjoint_method=None):
    if noise_size is None:
        noise_size = y0.shape[1]
    bm = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=(y0.shape[0], noise_size), dtype=torch.float64, device=device, entropy=entropy
    )
    if use_adjoint:
        return torchsde.sdeint_adjoint(
            sde, y0, ts, dt=dt, method=method, adjoint_method=adjoint_method or method, bm=bm
        )
    return torchsde.sdeint(sde, y0, ts, dt=dt, method=method, bm=bm)


def _terminal_loss(ys):
    loss = ys[-1].abs().pow(2).sum()
    loss.backward()
    return loss


def _collect_grads(module):
    return {name: param.grad.detach().clone() for name, param in module.named_parameters()}


def _relative_error(reference, test, min_denominator=1e-8):
    scale = reference.abs().max().clamp(min=min_denominator)
    return (reference - test).abs().max() / scale


def test_complex_adjoint_matches_backprop_complex_params():
    dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOULearnable()
    ys_bp = _run_solver(sde_bp, y0, ts, dt, method="euler", entropy=42)
    loss_bp = _terminal_loss(ys_bp)
    grad_alpha_bp = sde_bp.alpha.grad.detach().clone()
    grad_sigma_bp = sde_bp.sigma.grad.detach().clone()

    sde_adj = _ComplexOULearnable()
    _copy_parameters(sde_bp, sde_adj)
    ys_adj = _run_solver(
        sde_adj, y0, ts, dt, method="euler", entropy=42, use_adjoint=True, adjoint_method="euler"
    )
    loss_adj = _terminal_loss(ys_adj)
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

    assert _relative_error(grad_alpha_bp, grad_alpha_adj, min_denominator=1e-12).item() < 0.03
    assert _relative_error(grad_sigma_bp, grad_sigma_adj, min_denominator=1e-12).item() < 0.03


def test_complex_adjoint_matches_backprop_real_params():
    dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOURealParams()
    ys_bp = _run_solver(sde_bp, y0, ts, dt, method="euler", entropy=99)
    loss_bp = _terminal_loss(ys_bp)
    grads_bp = _collect_grads(sde_bp)

    sde_adj = _ComplexOURealParams()
    _copy_parameters(sde_bp, sde_adj)
    ys_adj = _run_solver(
        sde_adj, y0, ts, dt, method="euler", entropy=99, use_adjoint=True, adjoint_method="euler"
    )
    loss_adj = _terminal_loss(ys_adj)
    grads_adj = _collect_grads(sde_adj)

    torch.testing.assert_close(ys_bp, ys_adj, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(loss_bp, loss_adj, rtol=1e-10, atol=1e-10)

    for name, grad_bp in grads_bp.items():
        grad_adj = grads_adj[name]
        assert grad_adj.dtype == grad_bp.dtype
        assert not grad_adj.is_complex()
        assert torch.isfinite(grad_adj).all()
        assert _relative_error(grad_bp, grad_adj).item() < 0.05


def test_complex_adjoint_stratonovich_midpoint():
    dt, ts, y0 = _setup_problem()

    sde_bp = _ComplexOUStratLearnable()
    ys_bp = _run_solver(sde_bp, y0, ts, dt, method="midpoint", entropy=77)
    loss_bp = _terminal_loss(ys_bp)
    grad_alpha_bp = sde_bp.alpha.grad.detach().clone()

    sde_adj = _ComplexOUStratLearnable()
    _copy_parameters(sde_bp, sde_adj)
    ys_adj = _run_solver(
        sde_adj, y0, ts, dt, method="midpoint", entropy=77, use_adjoint=True, adjoint_method="midpoint"
    )
    loss_adj = _terminal_loss(ys_adj)
    grad_alpha_adj = sde_adj.alpha.grad.detach().clone()

    torch.testing.assert_close(ys_bp, ys_adj, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(loss_bp, loss_adj, rtol=1e-10, atol=1e-10)
    assert _relative_error(grad_alpha_bp, grad_alpha_adj, min_denominator=1e-12).item() < 0.03


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

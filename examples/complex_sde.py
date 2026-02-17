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

"""Minimal complex-valued example: compare sdeint vs sdeint_adjoint gradients."""

import torch
import torchsde


class ComplexOU(torch.nn.Module):
    """dY = -alpha * Y dt + sigma * dW, with complex alpha/sigma."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(
            torch.tensor([0.5 + 0.3j, 1.0 + 0.0j], dtype=torch.complex128)
        )
        self.sigma = torch.nn.Parameter(
            torch.tensor([0.5 + 0.2j, 0.3 + 0.0j], dtype=torch.complex128)
        )

    def f(self, t, y):
        return -self.alpha * y

    def g(self, t, y):
        return self.sigma.unsqueeze(0).expand_as(y)


def _compute_grads(use_adjoint, y0, ts, dt, entropy):
    sde = ComplexOU()
    bm = torchsde.ComplexBrownian(
        t0=ts[0], t1=ts[-1], size=y0.shape, dtype=torch.float64, entropy=entropy
    )

    if use_adjoint:
        ys = torchsde.sdeint_adjoint(
            sde,
            y0,
            ts,
            dt=dt,
            method="euler",
            adjoint_method="euler",
            bm=bm,
        )
    else:
        ys = torchsde.sdeint(sde, y0, ts, dt=dt, method="euler", bm=bm)

    loss = ys[-1].abs().pow(2).mean()
    loss.backward()
    return sde.alpha.grad.detach(), sde.sigma.grad.detach(), loss.item()


def main():
    torch.manual_seed(1234)

    batch_size = 64
    d = 2
    t0, t1 = 0.0, 0.5
    n_steps = 200
    dt = (t1 - t0) / n_steps

    ts = torch.linspace(t0, t1, n_steps + 1, dtype=torch.float64)
    y0 = torch.full((batch_size, d), 1.0 + 0.5j, dtype=torch.complex128)

    grad_alpha_bp, grad_sigma_bp, loss_bp = _compute_grads(
        use_adjoint=False, y0=y0, ts=ts, dt=dt, entropy=123
    )
    grad_alpha_adj, grad_sigma_adj, loss_adj = _compute_grads(
        use_adjoint=True, y0=y0, ts=ts, dt=dt, entropy=123
    )

    alpha_rel_err = (grad_alpha_bp - grad_alpha_adj).abs().max() / grad_alpha_bp.abs().max()
    sigma_rel_err = (grad_sigma_bp - grad_sigma_adj).abs().max() / grad_sigma_bp.abs().max()

    print("Loss (backprop): ", f"{loss_bp:.8f}")
    print("Loss (adjoint):  ", f"{loss_adj:.8f}")
    print("alpha grad (backprop):", grad_alpha_bp)
    print("alpha grad (adjoint): ", grad_alpha_adj)
    print("sigma grad (backprop):", grad_sigma_bp)
    print("sigma grad (adjoint): ", grad_sigma_adj)
    print("max relative error alpha:", f"{alpha_rel_err.item():.2e}")
    print("max relative error sigma:", f"{sigma_rel_err.item():.2e}")


if __name__ == "__main__":
    main()

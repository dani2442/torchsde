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

"""Complex-valued SDE example: Complex Ornstein-Uhlenbeck process
with complex Brownian motion.

Solves the SDE:
    dY = -alpha * Y dt + sigma * dW(t)

where Y in C (complex state), alpha in C, sigma in C, and
W(t) = W_1(t) + i*W_2(t) is a complex Brownian motion constructed from
two independent standard real Brownian motions W_1, W_2.

The analytical solution has:
    E[Y(t)] = Y(0) * exp(-alpha * t)
    Var[Y(t)] = |sigma|^2 / Re(alpha) * (1 - exp(-2 * Re(alpha) * t))

Note: The variance is twice what it would be with a real Brownian motion,
because E[|dW|^2] = E[|dW_1 + i*dW_2|^2] = 2*dt.

This example validates that torchsde correctly handles complex-valued initial
conditions, drift, and diffusion coefficients with complex Brownian motion.
"""

import torch
import torchsde


class ComplexOU(torch.nn.Module):
    """Complex Ornstein-Uhlenbeck process with diagonal noise.

    dY_i = -alpha_i * Y_i dt + sigma_i * dW_i(t)

    where alpha_i, sigma_i are complex parameters and W_i(t) = W_{1,i}(t) + i*W_{2,i}(t)
    are independent complex Brownian motions.
    """

    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, alpha, sigma):
        super().__init__()
        # alpha and sigma are complex tensors of shape (d,)
        self.register_buffer('alpha', alpha)
        self.register_buffer('sigma', sigma)

    def f(self, t, y):
        return -self.alpha * y

    def g(self, t, y):
        # Diffusion: sigma broadcast to (batch, d). The Brownian motion is
        # complex-valued (W_1 + i*W_2), so g * dW is complex.
        return self.sigma.unsqueeze(0).expand_as(y)


def main():
    torch.manual_seed(42)

    # --- Problem setup ---
    d = 4                   # State dimension
    batch_size = 8192       # Large batch for Monte Carlo statistics
    t0, t1 = 0.0, 2.0
    num_steps = 200
    dt = (t1 - t0) / num_steps

    # Complex parameters
    alpha = torch.tensor([0.5 + 0.3j, 1.0 + 0.0j, 0.3 - 0.5j, 0.8 + 0.2j])
    sigma = torch.tensor([0.5 + 0.2j, 0.3 + 0.0j, 0.4 - 0.1j, 0.2 + 0.3j])

    # Complex initial condition
    y0 = torch.full((batch_size, d), 1.0 + 0.5j, dtype=torch.complex128)

    ts = torch.linspace(t0, t1, num_steps + 1)

    sde = ComplexOU(alpha=alpha, sigma=sigma)

    # --- Solve the SDE ---
    print("Solving complex Ornstein-Uhlenbeck SDE with complex BM...")
    print(f"  State dimension: {d}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time interval: [{t0}, {t1}]")
    print(f"  dt: {dt:.4f}")
    print(f"  Complex BM: W = W_1 + i*W_2")
    print()

    ys = torchsde.sdeint(sde, y0, ts, dt=dt, method='euler')
    # ys has shape (num_steps+1, batch_size, d)

    print(f"  Output shape: {ys.shape}")
    print(f"  Output dtype: {ys.dtype}")
    assert ys.is_complex(), "Output should be complex-valued!"
    print()

    # --- Validate against analytical solution ---
    # At terminal time t1:
    #   E[Y(t1)] = y0 * exp(-alpha * t1)
    #   Var[Y(t1)] = |sigma|^2 / Re(alpha) * (1 - exp(-2 * Re(alpha) * t1))
    #
    # Note: The variance has a factor |sigma|^2 / Re(alpha) instead of
    # |sigma|^2 / (2*Re(alpha)) because E[|dW|^2] = 2*dt for complex BM.

    y_terminal = ys[-1]  # (batch_size, d)

    # Analytical mean (complex)
    analytical_mean = y0[0] * torch.exp(-alpha * t1)

    # Simulated mean
    simulated_mean = y_terminal.mean(dim=0)

    print("=== Terminal time statistics (t = {:.1f}) ===".format(t1))
    print()
    print("Mean (analytical vs simulated):")
    for i in range(d):
        print(f"  dim {i}: analytical = {analytical_mean[i]:.4f}, "
              f"simulated = {simulated_mean[i]:.4f}, "
              f"error = {abs(analytical_mean[i] - simulated_mean[i]):.4f}")

    # Analytical variance (real-valued: E[|Y - E[Y]|^2])
    # With complex BM: Var = |sigma|^2 / Re(alpha) * (1 - exp(-2*Re(alpha)*t))
    analytical_var = (torch.abs(sigma) ** 2 / alpha.real *
                      (1 - torch.exp(-2 * alpha.real * t1)))

    # Simulated variance: E[|Y - E[Y]|^2]
    centered = y_terminal - simulated_mean.unsqueeze(0)
    simulated_var = (torch.abs(centered) ** 2).mean(dim=0)

    print()
    print("Variance (analytical vs simulated):")
    for i in range(d):
        print(f"  dim {i}: analytical = {analytical_var[i].item():.4f}, "
              f"simulated = {simulated_var[i].item():.4f}, "
              f"error = {abs(analytical_var[i].item() - simulated_var[i].item()):.4f}")

    # --- Assertions ---
    mean_tol = 0.05
    var_tol = 0.08

    mean_error = torch.abs(analytical_mean - simulated_mean).max().item()
    var_error = torch.abs(analytical_var - simulated_var).max().item()

    print()
    print(f"Max mean error: {mean_error:.4f} (tolerance: {mean_tol})")
    print(f"Max variance error: {var_error:.4f} (tolerance: {var_tol})")

    assert mean_error < mean_tol, f"Mean error {mean_error:.4f} exceeds tolerance {mean_tol}"
    assert var_error < var_tol, f"Variance error {var_error:.4f} exceeds tolerance {var_tol}"

    print()
    print("All assertions passed! Complex SDE integration is working correctly.")

    # --- Also test with Stratonovich midpoint method ---
    print()
    print("--- Testing Stratonovich midpoint method ---")

    class ComplexOUStratonovich(torch.nn.Module):
        noise_type = 'diagonal'
        sde_type = 'stratonovich'

        def __init__(self, alpha, sigma):
            super().__init__()
            self.register_buffer('alpha', alpha)
            self.register_buffer('sigma', sigma)

        def f(self, t, y):
            return -self.alpha * y

        def g(self, t, y):
            return self.sigma.unsqueeze(0).expand_as(y)

    sde_strat = ComplexOUStratonovich(alpha=alpha, sigma=sigma)
    ys_strat = torchsde.sdeint(sde_strat, y0, ts, dt=dt, method='midpoint')

    print(f"  Output shape: {ys_strat.shape}")
    print(f"  Output dtype: {ys_strat.dtype}")
    assert ys_strat.is_complex(), "Stratonovich output should be complex-valued!"

    strat_mean = ys_strat[-1].mean(dim=0)
    strat_mean_error = torch.abs(analytical_mean - strat_mean).max().item()
    print(f"  Max mean error: {strat_mean_error:.4f}")
    assert strat_mean_error < mean_tol, f"Stratonovich mean error exceeds tolerance"
    print("  Stratonovich test passed!")

    # --- Test with general noise type ---
    print()
    print("--- Testing general noise type ---")

    m = 3  # Brownian motion dimension (different from state dimension)

    class ComplexGeneralSDE(torch.nn.Module):
        noise_type = 'general'
        sde_type = 'ito'

        def __init__(self, d, m):
            super().__init__()
            # Complex diffusion matrix
            G_real = torch.randn(d, m) * 0.3
            G_imag = torch.randn(d, m) * 0.3
            self.register_buffer('G', torch.complex(G_real, G_imag))

        def f(self, t, y):
            return -0.5 * y

        def g(self, t, y):
            # (batch, d, m) - constant diffusion
            return self.G.unsqueeze(0).expand(y.shape[0], -1, -1)

    sde_gen = ComplexGeneralSDE(d=d, m=m)
    y0_gen = torch.full((batch_size, d), 1.0 + 0.5j, dtype=torch.complex128)
    ys_gen = torchsde.sdeint(sde_gen, y0_gen, ts, dt=dt, method='euler')

    print(f"  Output shape: {ys_gen.shape}")
    print(f"  Output dtype: {ys_gen.dtype}")
    assert ys_gen.is_complex(), "General noise output should be complex-valued!"
    print("  General noise test passed!")

    print()
    print("=" * 50)
    print("All complex SDE tests passed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()

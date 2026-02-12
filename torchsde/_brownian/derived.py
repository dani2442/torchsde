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

import torch

from . import brownian_base
from . import brownian_interval
from ..types import Optional, Scalar, Tensor, Tuple, Union


class ReverseBrownian(brownian_base.BaseBrownian):
    def __init__(self, base_brownian):
        super(ReverseBrownian, self).__init__()
        self.base_brownian = base_brownian

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        # Whether or not to negate the statistics depends on the return value of the adjoint SDE. Currently, the adjoint
        # returns negated drift and diffusion, so we don't negate here.
        return self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)

    def __repr__(self):
        return f"{self.__class__.__name__}(base_brownian={self.base_brownian})"

    @property
    def dtype(self):
        return self.base_brownian.dtype

    @property
    def device(self):
        return self.base_brownian.device

    @property
    def shape(self):
        return self.base_brownian.shape

    @property
    def levy_area_approximation(self):
        return self.base_brownian.levy_area_approximation


class BrownianPath(brownian_base.BaseBrownian):
    """Brownian path, storing every computed value.

    Useful for speed, when memory isn't a concern.

    To use:
    >>> bm = BrownianPath(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self, t0: Scalar, w0: Tensor, window_size: int = 8):
        """Initialize Brownian path.
        Arguments:
            t0: Initial time.
            w0: Initial state.
            window_size: Unused; deprecated.
        """
        t1 = t0 + 1
        self._w0 = w0
        self._interval = brownian_interval.BrownianInterval(t0=t0, t1=t1, size=w0.shape, dtype=w0.dtype,
                                                            device=w0.device, cache_size=None)
        super(BrownianPath, self).__init__()

    def __call__(self, t, tb=None, return_U=False, return_A=False):
        # Deliberately called t rather than ta, for backward compatibility
        out = self._interval(t, tb, return_U=return_U, return_A=return_A)
        if tb is None and not return_U and not return_A:
            out = out + self._w0
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self._interval})"

    @property
    def dtype(self):
        return self._interval.dtype

    @property
    def device(self):
        return self._interval.device

    @property
    def shape(self):
        return self._interval.shape

    @property
    def levy_area_approximation(self):
        return self._interval.levy_area_approximation


class BrownianTree(brownian_base.BaseBrownian):
    """Brownian tree with fixed entropy.

    Useful when the map from entropy -> Brownian motion shouldn't depend on the
    locations and order of the query points. (As the usual BrownianInterval
    does - note that BrownianTree is slower as a result though.)

    To use:
    >>> bm = BrownianTree(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    def __init__(self, t0: Scalar,
                 w0: Tensor,
                 t1: Optional[Scalar] = None,
                 w1: Optional[Tensor] = None,
                 entropy: Optional[int] = None,
                 tol: float = 1e-6,
                 pool_size: int = 24,
                 cache_depth: int = 9,
                 safety: Optional[float] = None):
        """Initialize the Brownian tree.

        The random value generation process exploits the parallel random number paradigm and uses
        `numpy.random.SeedSequence`. The default generator is PCG64 (used by `default_rng`).

        Arguments:
            t0: Initial time.
            w0: Initial state.
            t1: Terminal time.
            w1: Terminal state.
            entropy: Global seed, defaults to `None` for random entropy.
            tol: Error tolerance before the binary search is terminated; the search depth ~ log2(tol).
            pool_size: Size of the pooled entropy. This parameter affects the query speed significantly.
            cache_depth: Unused; deprecated.
            safety: Unused; deprecated.
        """

        if t1 is None:
            t1 = t0 + 1
        if w1 is None:
            W = None
        else:
            W = w1 - w0
        self._w0 = w0
        self._interval = brownian_interval.BrownianInterval(t0=t0,
                                                            t1=t1,
                                                            size=w0.shape,
                                                            dtype=w0.dtype,
                                                            device=w0.device,
                                                            entropy=entropy,
                                                            tol=tol,
                                                            pool_size=pool_size,
                                                            halfway_tree=True,
                                                            W=W)
        super(BrownianTree, self).__init__()

    def __call__(self, t, tb=None, return_U=False, return_A=False):
        # Deliberately called t rather than ta, for backward compatibility
        out = self._interval(t, tb, return_U=return_U, return_A=return_A)
        if tb is None and not return_U and not return_A:
            out = out + self._w0
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self._interval})"

    @property
    def dtype(self):
        return self._interval.dtype

    @property
    def device(self):
        return self._interval.device

    @property
    def shape(self):
        return self._interval.shape

    @property
    def levy_area_approximation(self):
        return self._interval.levy_area_approximation


class ComplexBrownian(brownian_base.BaseBrownian):
    """Complex Brownian motion W = W_1 + i * W_2.

    Constructs a complex-valued Brownian motion from two independent real-valued
    BrownianInterval objects. Given d-dimensional real Brownian motions W_1, W_2,
    the complex Brownian motion is defined component-wise as:

        W_k(t) = W_{1,k}(t) + i * W_{2,k}(t),   k = 1, ..., d.

    Each component W_k has:
        E[W_k(t)] = 0
        E[|W_k(t)|^2] = E[W_{1,k}(t)^2] + E[W_{2,k}(t)^2] = 2t

    The increments dW_k = dW_{1,k} + i * dW_{2,k} satisfy:
        E[dW_k] = 0
        E[|dW_k|^2] = 2 dt

    To use:
    >>> bm = ComplexBrownian(t0=0.0, t1=1.0, size=(4, 2))
    >>> dW = bm(0.0, 0.5)  # complex tensor of shape (4, 2)
    """

    def __init__(self, t0: Scalar, t1: Scalar,
                 size: Tuple[int, ...],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs):
        """Initialize complex Brownian motion from two independent real BrownianIntervals.

        Args:
            t0: Initial time.
            t1: Terminal time.
            size: Shape of each Brownian sample (batch, d).
            dtype: The *real* dtype for the underlying intervals (e.g. torch.float64).
                   If a complex dtype is passed, the corresponding real dtype is used.
            device: Device for the Brownian samples.
            **kwargs: Additional keyword arguments passed to BrownianInterval
                      (e.g. levy_area_approximation, entropy, dt, etc.).
        """
        super(ComplexBrownian, self).__init__()

        # Resolve real dtype from potentially complex input dtype.
        if dtype is not None and dtype.is_complex:
            _real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        else:
            _real_dtype = dtype

        # Ensure the two intervals use different entropy/seeds for independence.
        import numpy as np
        entropy1 = kwargs.pop('entropy', None)
        if entropy1 is None:
            entropy1 = np.random.randint(0, 2 ** 31 - 1)
        # Derive a deterministically different entropy for the second interval.
        entropy2 = (entropy1 + 1) % (2 ** 31)

        self._bm_real = brownian_interval.BrownianInterval(
            t0=t0, t1=t1, size=size, dtype=_real_dtype, device=device,
            entropy=entropy1, **kwargs
        )
        self._bm_imag = brownian_interval.BrownianInterval(
            t0=t0, t1=t1, size=size, dtype=_real_dtype, device=device,
            entropy=entropy2, **kwargs
        )
        self._size = size

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        # Delegate to both underlying real BrownianIntervals.
        out_real = self._bm_real(ta, tb, return_U=return_U, return_A=return_A)
        out_imag = self._bm_imag(ta, tb, return_U=return_U, return_A=return_A)

        def _to_complex(re, im):
            return re + 1j * im

        if isinstance(out_real, tuple):
            return tuple(_to_complex(r, i) for r, i in zip(out_real, out_imag))
        else:
            return _to_complex(out_real, out_imag)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"bm_real={self._bm_real}, "
                f"bm_imag={self._bm_imag})")

    @property
    def dtype(self):
        real_dt = self._bm_real.dtype
        if real_dt == torch.float32:
            return torch.complex64
        else:
            return torch.complex128

    @property
    def device(self):
        return self._bm_real.device

    @property
    def shape(self):
        return self._size

    @property
    def levy_area_approximation(self):
        return self._bm_real.levy_area_approximation


def brownian_interval_like(y: Tensor,
                           t0: Optional[Scalar] = 0.,
                           t1: Optional[Scalar] = 1.,
                           size: Optional[Tuple[int, ...]] = None,
                           dtype: Optional[torch.dtype] = None,
                           device: Optional[Union[str, torch.device]] = None,
                           **kwargs):
    """Returns a BrownianInterval object with the same size, device, and dtype as a given tensor."""
    size = y.shape if size is None else size
    dtype = y.dtype if dtype is None else dtype
    device = y.device if device is None else device
    return brownian_interval.BrownianInterval(t0=t0, t1=t1, size=size, dtype=dtype, device=device, **kwargs)

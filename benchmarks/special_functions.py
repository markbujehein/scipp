# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import ClassVar

import scipp as sc


class SphericalBessel:
    """
    Benchmark spherical Bessel functions for different sizes and orders.
    """

    params: ClassVar[list[list[int]]] = [[10**4, 10**6], [0, 5, 50]]
    param_names: ClassVar[list[str]] = ['size', 'order']

    def setup(self, size: int, order: int) -> None:
        self.x = sc.linspace('x', -100.0, 100.0, size)

    def time_spherical_jn(self, size: int, order: int) -> None:
        sc.spherical_jn(order, self.x)

    def time_spherical_yn(self, size: int, order: int) -> None:
        sc.spherical_yn(order, self.x)

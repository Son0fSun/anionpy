"""anionpy.lib -- the `numpy.lib.*` block (a.k.a. `numpy.lib` re-exports).

Two items live here: `NumpyVersion` and `Arrayterator`. Both are ported
VERBATIM from numpy 2.5.1's own installed source (`numpy/lib/_version.py`'s
`NumpyVersion`, `numpy/lib/_arrayterator_impl.py`'s `Arrayterator`), not
reimplemented from a description, because neither one touches array
arithmetic at all:

- `NumpyVersion` parses/compares dotted version STRINGS with `re` and plain
  int comparisons. There is no array, no ndarray, no numeric elementwise
  operation anywhere in it -- it is string/int bookkeeping, the same
  category of "Python control flow, not Python arithmetic-on-array-data"
  that `anionpy/testing.py`'s module docstring already carves out.
- `Arrayterator` is a lazy, buffered slicing iterator. Its entire body is
  index-arithmetic on Python `int`s (`start`/`stop`/`step` lists) plus
  `self.var[slice_]` -- delegating the actual sub-array extraction to
  whatever `__getitem__` the wrapped object provides. It never reads or
  computes an array ELEMENT itself; it only decides which slice to ask
  the wrapped array for next. Verified directly against `anionpy.ndarray`:
  `.shape`, `.ndim`, and tuple-of-`slice` `__getitem__` are all already
  Rust-backed and behave exactly as `Arrayterator` requires (see
  tests/differential/lib_cases.py).

One documented gap: numpy's `Arrayterator.flat` property (`yield from
block.flat`) depends on `ndarray.flat`, which `anionpy.ndarray` does not
implement (nothing in this task's scope adds it -- `ndarray.flat` is
ndarray's own surface, not `lib`'s, and editing `ionp-py/src/lib.rs`'s
`PyArray` impl is out of this task's file ownership). `lib.Arrayterator`
the CLASS is fully portable and is what `numpy_surface.json` actually
lists; `.flat` is a separate attribute access this task neither declares
nor tests. Iterating an `Arrayterator` wrapping an `anionpy.ndarray` (its
primary, documented use) works correctly; touching `.flat` on the result
raises `AttributeError`, exactly as it would for any other object numpy's
own `Arrayterator` wraps that lacks a `.flat` (e.g. a plain h5py dataset).
"""
from __future__ import annotations

import re
from functools import reduce
from operator import mul

__all__ = ["NumpyVersion", "Arrayterator"]


class NumpyVersion:
    """Parse and compare numpy-style version strings.

    Ported verbatim from `numpy.lib.NumpyVersion` (`numpy/lib/_version.py`).
    """

    __module__ = "anionpy.lib"

    def __init__(self, vstring):
        self.vstring = vstring
        ver_main = re.match(r'\d+\.\d+\.\d+', vstring)
        if not ver_main:
            raise ValueError("Not a valid numpy version string")

        self.version = ver_main.group()
        self.major, self.minor, self.bugfix = [int(x) for x in
            self.version.split('.')]
        if len(vstring) == ver_main.end():
            self.pre_release = 'final'
        else:
            alpha = re.match(r'a\d', vstring[ver_main.end():])
            beta = re.match(r'b\d', vstring[ver_main.end():])
            rc = re.match(r'rc\d', vstring[ver_main.end():])
            pre_rel = [m for m in [alpha, beta, rc] if m is not None]
            if pre_rel:
                self.pre_release = pre_rel[0].group()
            else:
                self.pre_release = ''

        self.is_devversion = bool(re.search(r'.dev', vstring))

    def _compare_version(self, other):
        """Compare major.minor.bugfix"""
        if self.major == other.major:
            if self.minor == other.minor:
                if self.bugfix == other.bugfix:
                    vercmp = 0
                elif self.bugfix > other.bugfix:
                    vercmp = 1
                else:
                    vercmp = -1
            elif self.minor > other.minor:
                vercmp = 1
            else:
                vercmp = -1
        elif self.major > other.major:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare_pre_release(self, other):
        """Compare alpha/beta/rc/final."""
        if self.pre_release == other.pre_release:
            vercmp = 0
        elif self.pre_release == 'final':
            vercmp = 1
        elif other.pre_release == 'final':
            vercmp = -1
        elif self.pre_release > other.pre_release:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare(self, other):
        if not isinstance(other, (str, NumpyVersion)):
            raise ValueError("Invalid object to compare with NumpyVersion.")

        if isinstance(other, str):
            other = NumpyVersion(other)

        vercmp = self._compare_version(other)
        if vercmp == 0:
            # Same x.y.z version, check for alpha/beta/rc
            vercmp = self._compare_pre_release(other)
            if vercmp == 0:
                # Same version and same pre-release, check if dev version
                if self.is_devversion is other.is_devversion:
                    vercmp = 0
                elif self.is_devversion:
                    vercmp = -1
                else:
                    vercmp = 1

        return vercmp

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __repr__(self):
        return f"NumpyVersion({self.vstring})"


class Arrayterator:
    """Buffered iterator for big arrays.

    Ported verbatim from `numpy.lib.Arrayterator`
    (`numpy/lib/_arrayterator_impl.py`), unpacking blocks of at most
    `buf_size` elements out of any object supporting multidimensional
    `__getitem__`/`.shape`/`.ndim` -- `anionpy.ndarray` included. See this
    module's docstring for the one documented gap (`.flat`).
    """

    __module__ = "anionpy.lib"

    def __init__(self, var, buf_size=None):
        self.var = var
        self.buf_size = buf_size

        self.start = [0 for dim in var.shape]
        self.stop = list(var.shape)
        self.step = [1 for dim in var.shape]

    def __getattr__(self, attr):
        return getattr(self.var, attr)

    def __getitem__(self, index):
        """Return a new arrayterator."""
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims - len(index))

        # Return a new arrayterator object.
        out = self.__class__(self.var, self.buf_size)
        for i, (start, stop, step, slice_) in enumerate(
                zip(self.start, self.stop, self.step, index)):
            out.start[i] = start + (slice_.start or 0)
            out.step[i] = step * (slice_.step or 1)
            out.stop[i] = start + (slice_.stop or stop - start)
            out.stop[i] = min(stop, out.stop[i])
        return out

    def __array__(self, dtype=None, copy=None):
        """Return corresponding data."""
        slice_ = tuple(slice(*t) for t in zip(
                self.start, self.stop, self.step))
        return self.var[slice_]

    @property
    def flat(self):
        """A 1-D flat iterator for Arrayterator objects.

        Requires the wrapped object to expose `.flat` itself (see this
        module's docstring -- `anionpy.ndarray` does not, currently).
        """
        for block in self:
            yield from block.flat

    @property
    def shape(self):
        """The shape of the array to be iterated over."""
        return tuple(((stop - start - 1) // step + 1) for start, stop, step in
                zip(self.start, self.stop, self.step))

    def __iter__(self):
        # Skip arrays with degenerate dimensions
        if [dim for dim in self.shape if dim <= 0]:
            return

        start = self.start[:]
        stop = self.stop[:]
        step = self.step[:]
        ndims = self.var.ndim

        while True:
            count = self.buf_size or reduce(mul, self.shape)

            # iterate over each dimension, looking for the
            # running dimension (ie, the dimension along which
            # the blocks will be built from)
            rundim = 0
            for i in range(ndims - 1, -1, -1):
                # if count is zero we ran out of elements to read
                # along higher dimensions, so we read only a single position
                if count == 0:
                    stop[i] = start[i] + 1
                elif count <= self.shape[i]:
                    # limit along this dimension
                    stop[i] = start[i] + count * step[i]
                    rundim = i
                else:
                    # read everything along this dimension
                    stop[i] = self.stop[i]
                stop[i] = min(self.stop[i], stop[i])
                count = count // self.shape[i]

            # yield a block
            slice_ = tuple(slice(*t) for t in zip(start, stop, step))
            yield self.var[slice_]

            # Update start position, taking care of overflow to
            # other dimensions
            start[rundim] = stop[rundim]  # start where we stopped
            for i in range(ndims - 1, 0, -1):
                if start[i] >= self.stop[i]:
                    start[i] = self.start[i]
                    start[i - 1] += self.step[i - 1]
            if start[0] >= self.stop[0]:
                return

"""Out-of-corpus probe for median/percentile/quantile/nan* -- NOT part of
the shipped test suite (underscore-prefixed, invoked manually). Sweeps
axis/keepdims/method/dtype/empty/all-nan/0d against real numpy, comparing
byte-level (tobytes + dtype str + shape). Not imported by run.py.
"""
import sys, traceback
import numpy as np
import anionpy

METHODS = ["inverted_cdf","averaged_inverted_cdf","closest_observation",
           "interpolated_inverted_cdf","hazen","weibull","linear",
           "median_unbiased","normal_unbiased","lower","higher","midpoint","nearest"]

DTYPES = [np.bool_, np.int8, np.int16, np.int32, np.int64,
          np.uint8, np.uint16, np.uint32, np.uint64,
          np.float16, np.float32, np.float64]

total = 0
mismatches = []
errors_diff = []

def ionp_to_np(x):
    return np.asarray(x)

def compare(label, fn_ionp, fn_np):
    global total
    total += 1
    try:
        r_i = fn_ionp()
        i_ok = True
    except BaseException as e:
        r_i = e
        i_ok = False
    try:
        r_n = fn_np()
        n_ok = True
    except BaseException as e:
        r_n = e
        n_ok = False

    if i_ok != n_ok:
        mismatches.append((label, "presence-mismatch", repr(r_i), repr(r_n)))
        return
    if not i_ok:
        # both raised -- check exception type + message
        if type(r_i) is not type(r_n) and not (isinstance(r_i, type(r_n)) or isinstance(r_n, type(r_i))):
            errors_diff.append((label, type(r_i).__name__, type(r_n).__name__, str(r_i), str(r_n)))
        elif str(r_i) != str(r_n):
            errors_diff.append((label, type(r_i).__name__, type(r_n).__name__, str(r_i), str(r_n)))
        return
    a = ionp_to_np(r_i)
    b = np.asarray(r_n)
    if a.shape != b.shape or str(a.dtype) != str(b.dtype):
        mismatches.append((label, "shape/dtype", f"{a.shape}/{a.dtype}", f"{b.shape}/{b.dtype}"))
        return
    eq_nan = a.dtype.kind in "fc"
    if not np.array_equal(a, b, equal_nan=eq_nan):
        mismatches.append((label, "value", a.tobytes()[:200], b.tobytes()[:200]))

rng = np.random.default_rng(12345)

for dt in DTYPES:
    for shape in [(1,), (2,), (3,), (5,), (2,3), (3,4,2), (0,), (2,0,3)]:
        if dt == np.bool_:
            arr = rng.integers(0,2,size=shape).astype(bool)
        elif np.issubdtype(dt, np.integer):
            arr = rng.integers(-50,50,size=shape).astype(dt)
        else:
            arr = (rng.standard_normal(size=shape)*10).astype(dt)
        ndim = arr.ndim
        axes = [None] + list(range(-ndim, ndim)) if ndim>0 else [None]
        for axis in axes:
            for keepdims in [False, True]:
                lbl = f"median dt={dt.__name__} shape={shape} axis={axis} kd={keepdims}"
                compare(lbl, lambda a=arr,ax=axis,k=keepdims: anionpy.median(a,axis=ax,keepdims=k),
                              lambda a=arr,ax=axis,k=keepdims: np.median(a,axis=ax,keepdims=k))
                lbl2 = f"nanmedian dt={dt.__name__} shape={shape} axis={axis} kd={keepdims}"
                compare(lbl2, lambda a=arr,ax=axis,k=keepdims: anionpy.nanmedian(a,axis=ax,keepdims=k),
                               lambda a=arr,ax=axis,k=keepdims: np.nanmedian(a,axis=ax,keepdims=k))
                for m in METHODS:
                    for q, qlabel in [(0.5,"scalar"), ([0.0,0.25,0.5,0.75,1.0],"array"), (0.0,"q0"), (1.0,"q1")]:
                        lbl3 = f"quantile dt={dt.__name__} shape={shape} axis={axis} kd={keepdims} m={m} q={qlabel}"
                        compare(lbl3, lambda a=arr,ax=axis,k=keepdims,mm=m,qq=q: anionpy.quantile(a,qq,axis=ax,method=mm,keepdims=k),
                                       lambda a=arr,ax=axis,k=keepdims,mm=m,qq=q: np.quantile(a,qq,axis=ax,method=mm,keepdims=k))
                        lbl4 = f"nanquantile dt={dt.__name__} shape={shape} axis={axis} kd={keepdims} m={m} q={qlabel}"
                        compare(lbl4, lambda a=arr,ax=axis,k=keepdims,mm=m,qq=q: anionpy.nanquantile(a,qq,axis=ax,method=mm,keepdims=k),
                                       lambda a=arr,ax=axis,k=keepdims,mm=m,qq=q: np.nanquantile(a,qq,axis=ax,method=mm,keepdims=k))
                        # percentile scale
                        pq = [v*100 for v in q] if isinstance(q,list) else q*100
                        lbl5 = f"percentile dt={dt.__name__} shape={shape} axis={axis} kd={keepdims} m={m} q={qlabel}"
                        compare(lbl5, lambda a=arr,ax=axis,k=keepdims,mm=m,qq=pq: anionpy.percentile(a,qq,axis=ax,method=mm,keepdims=k),
                                       lambda a=arr,ax=axis,k=keepdims,mm=m,qq=pq: np.percentile(a,qq,axis=ax,method=mm,keepdims=k))

# all-NaN slices (float dtypes only)
for dt in [np.float16, np.float32, np.float64]:
    for shape in [(4,), (2,3)]:
        arr = np.full(shape, np.nan, dtype=dt)
        compare(f"allnan median dt={dt.__name__} shape={shape}", lambda a=arr: anionpy.median(a), lambda a=arr: np.median(a))
        compare(f"allnan nanmedian dt={dt.__name__} shape={shape}", lambda a=arr: anionpy.nanmedian(a), lambda a=arr: np.nanmedian(a))
        for m in METHODS:
            compare(f"allnan quantile dt={dt.__name__} shape={shape} m={m}", lambda a=arr,mm=m: anionpy.quantile(a,[0.1,0.9],method=mm), lambda a=arr,mm=m: np.quantile(a,[0.1,0.9],method=mm))
            compare(f"allnan nanquantile dt={dt.__name__} shape={shape} m={m}", lambda a=arr,mm=m: anionpy.nanquantile(a,[0.1,0.9],method=mm), lambda a=arr,mm=m: np.nanquantile(a,[0.1,0.9],method=mm))

# partial-NaN 2D rows (mixed)
mixed = np.array([[1.,2.,np.nan],[np.nan,np.nan,np.nan],[4.,5.,6.]])
for m in METHODS:
    compare(f"mixed-nan quantile m={m}", lambda mm=m: anionpy.quantile(mixed,[0.25,0.75],axis=1,method=mm), lambda mm=m: np.quantile(mixed,[0.25,0.75],axis=1,method=mm))
    compare(f"mixed-nan nanquantile m={m}", lambda mm=m: anionpy.nanquantile(mixed,[0.25,0.75],axis=1,method=mm), lambda mm=m: np.nanquantile(mixed,[0.25,0.75],axis=1,method=mm))
compare("mixed-nan median", lambda: anionpy.median(mixed,axis=1), lambda: np.median(mixed,axis=1))
compare("mixed-nan nanmedian", lambda: anionpy.nanmedian(mixed,axis=1), lambda: np.nanmedian(mixed,axis=1))

# 0d arrays
for dt in DTYPES:
    v = np.array(5, dtype=dt) if dt != np.bool_ else np.array(True)
    compare(f"0d median dt={dt.__name__}", lambda a=v: anionpy.median(a), lambda a=v: np.median(a))
    for m in METHODS:
        compare(f"0d quantile dt={dt.__name__} m={m}", lambda a=v,mm=m: anionpy.quantile(a,0.5,method=mm), lambda a=v,mm=m: np.quantile(a,0.5,method=mm))

# error triggers
def err_case(label, fn_ionp, fn_np):
    compare(label, fn_ionp, fn_np)

err_case("percentile out of range >100", lambda: anionpy.percentile(np.array([1.,2.,3.]), 150), lambda: np.percentile(np.array([1.,2.,3.]),150))
err_case("percentile out of range <0", lambda: anionpy.percentile(np.array([1.,2.,3.]), -1), lambda: np.percentile(np.array([1.,2.,3.]),-1))
err_case("quantile out of range >1", lambda: anionpy.quantile(np.array([1.,2.,3.]), 1.5), lambda: np.quantile(np.array([1.,2.,3.]),1.5))
err_case("quantile out of range <0", lambda: anionpy.quantile(np.array([1.,2.,3.]), -0.1), lambda: np.quantile(np.array([1.,2.,3.]),-0.1))
err_case("bool linear quantile", lambda: anionpy.quantile(np.array([True,False]), 0.5, method="linear"), lambda: np.quantile(np.array([True,False]),0.5,method="linear"))
err_case("complex percentile", lambda: anionpy.percentile(np.array([1+1j,2+2j]), 50), lambda: np.percentile(np.array([1+1j,2+2j]),50))
err_case("complex quantile inverted_cdf", lambda: anionpy.quantile(np.array([1+1j,2+2j]), 0.5, method="inverted_cdf"), lambda: np.quantile(np.array([1+1j,2+2j]),0.5,method="inverted_cdf"))
err_case("complex median (should work)", lambda: anionpy.median(np.array([1+1j,2+2j,3+3j])), lambda: np.median(np.array([1+1j,2+2j,3+3j])))
err_case("empty percentile", lambda: anionpy.percentile(np.array([],dtype=np.float64), 50), lambda: np.percentile(np.array([],dtype=np.float64),50))
err_case("empty quantile array-q", lambda: anionpy.quantile(np.array([],dtype=np.float64), [0.1,0.9]), lambda: np.quantile(np.array([],dtype=np.float64),[0.1,0.9]))
err_case("empty median", lambda: anionpy.median(np.array([],dtype=np.float64)), lambda: np.median(np.array([],dtype=np.float64)))
err_case("bad method name", lambda: anionpy.quantile(np.array([1.,2.,3.]), 0.5, method="bogus"), lambda: np.quantile(np.array([1.,2.,3.]),0.5,method="bogus"))
err_case("axis out of bounds", lambda: anionpy.median(np.array([1.,2.,3.]), axis=5), lambda: np.median(np.array([1.,2.,3.]),axis=5))
err_case("q ndim2 error", lambda: anionpy.quantile(np.array([1.,2.,3.]), np.array([[0.1,0.2],[0.3,0.4]])), lambda: np.quantile(np.array([1.,2.,3.]),np.array([[0.1,0.2],[0.3,0.4]])))

# out= param
out_i = anionpy.array(np.empty((), dtype=np.float64))
out_n = np.empty((), dtype=np.float64)
r_i = anionpy.percentile(np.array([1.,2.,3.]), 50.0, out=out_i)
r_n = np.percentile(np.array([1.,2.,3.]), 50.0, out=out_n)
compare("out param", lambda: out_i, lambda: out_n)

# large 4d with tuple negative axis single-int combos already covered; keepdims+axis None
big = rng.standard_normal((2,3,4,5))
for m in METHODS:
    compare(f"4d axisNone m={m}", lambda mm=m: anionpy.quantile(big,[0.3,0.6],method=mm), lambda mm=m: np.quantile(big,[0.3,0.6],method=mm))

print(f"\nTOTAL PROBES: {total}")
print(f"MISMATCHES: {len(mismatches)}")
for m in mismatches[:40]:
    print("  MISMATCH:", m)
print(f"ERROR-TYPE/MESSAGE DIFFS: {len(errors_diff)}")
for e in errors_diff[:40]:
    print("  ERRDIFF:", e)

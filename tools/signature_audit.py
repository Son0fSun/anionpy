#!/usr/bin/env python3
"""Signature audit: find declared-green items that reject parameters numpy accepts.

The differential corpus tests VALUES on the DEFAULT CALL FORM. It is therefore
structurally blind to a missing PARAMETER -- an item can ignore half its numpy
signature and still be graded `pass`. This script closes that blind spot from
the other side: it compares `inspect.signature` for every passing item against
numpy's and reports the difference.

    python3 tests/differential/run.py --out /tmp/rep.json
    python3 tools/signature_audit.py /tmp/rep.json

A suspect here is NOT automatically a bug -- some differences are only a
positional parameter spelled differently (numpy's `np.array(object, ...)` vs
anionpy's own name for the same slot). Every suspect must be confirmed by actually
CALLING it with that parameter before it is treated as real. When first run
(2026-08-01) this reported 23 suspects, of which 16 were confirmed real by
probe: eye/order, array/ndmin+copy+order+subok, asarray/order+copy,
reshape/copy, linspace/axis, zeros/like, zeros_like/subok, broadcast_to/subok,
copy/subok, linalg.svd/hermitian, linalg.trace/dtype.
"""
import numpy as np, anionpy, inspect, json, sys
rep=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/tmp/rep.json'))
declared=[k for k,v in rep.items() if (v.get("verdict") if isinstance(v,dict) else v)=="pass"]
missing=[]; checked=0
def resolve(mod, name):
    o=mod
    for p in name.split("."):
        o=getattr(o,p,None)
        if o is None: return None
    return o
for name in sorted(declared):
    if name.startswith("ndarray."):
        npo=resolve(np,"ndarray."+name.split(".",1)[1]); ipo=resolve(anionpy,"ndarray."+name.split(".",1)[1])
    else:
        npo=resolve(np,name); ipo=resolve(anionpy,name)
    if npo is None or ipo is None: continue
    if isinstance(npo,np.ufunc): continue
    try: nsig=inspect.signature(npo)
    except (ValueError,TypeError): continue
    try: isig=inspect.signature(ipo)
    except (ValueError,TypeError):
        missing.append((name,"anionpy sig unintrospectable",sorted(nsig.parameters))); continue
    checked+=1
    npar=set(nsig.parameters); ipar=set(isig.parameters)
    gap=sorted(p for p in npar-ipar if p not in ("args","kwargs"))
    if gap and not any(p.kind==p.VAR_KEYWORD for p in isig.parameters.values()):
        missing.append((name,"missing params",gap))
print("declared=%d introspectable=%d suspects=%d"%(len(declared),checked,len(missing)))
for n,w,g in missing: print("  %-28s %-26s %s"%(n,w,g))

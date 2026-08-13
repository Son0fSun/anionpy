"""Coverage declarations for the `numpy.testing` block.

Split out of `anionpy/__init__.py` on 2026-08-01. See
`reports/ionp-throughput-bottleneck-2026-08-01.md`: every declaration used to
live in one dict in `__init__.py`, which meant any two agents declaring work
had to be serialised through one file -- and five separate blocks of finished,
tested, committed work went uncounted because wiring them in was somebody
else's file to touch. One block per module, merged with collision detection.

The value is the ledger state string ("exact" / "ion"). The COMMENTS ARE THE
EVIDENCE and are load-bearing: an entry without a recorded reason is not
better than no entry. Do not add a key here without the measurement that
justifies it. Moved verbatim -- no declaration was added, removed or altered
by the split itself.
"""

TESTING_STATE = {

    # numpy.testing block, declared 2026-08-01. All 47 verified twice:
    #   * differential suite : 47/47 pass, bit-exact, no tolerance claimed
    #   * falsifiability     : 47/47 CAUGHT -- each test provably fails when a
    #     deliberately wrong implementation is substituted (tools/falsifiability.py,
    #     instrument itself validated 21/21 against known-bad snapshot b784819)
    # Declared only after both, because 21 of these once passed while being
    # unfalsifiable -- a green tick that graded a wrong implementation as correct.
    "testing.BLAS_SUPPORTS_FPE": "exact",
    "testing.HAS_LAPACK64": "exact",
    "testing.HAS_REFCOUNT": "exact",
    "testing.IS_64BIT": "exact",
    "testing.IS_EDITABLE": "exact",
    "testing.IS_INSTALLED": "exact",
    "testing.IS_MUSL": "exact",
    "testing.IS_PYPY": "exact",
    "testing.IS_PYSTON": "exact",
    "testing.IS_WASM": "exact",
    "testing.IgnoreException": "exact",
    "testing.KnownFailureException": "exact",
    "testing.NOGIL_BUILD": "exact",
    "testing.NUMPY_ROOT": "exact",
    "testing.SkipTest": "exact",
    "testing.TestCase": "exact",
    "testing.assert_": "exact",
    "testing.assert_allclose": "exact",
    "testing.assert_almost_equal": "exact",
    "testing.assert_approx_equal": "exact",
    "testing.assert_array_almost_equal": "exact",
    "testing.assert_array_equal": "exact",
    "testing.assert_array_less": "exact",
    "testing.assert_equal": "exact",
    "testing.assert_no_gc_cycles": "exact",
    "testing.assert_no_warnings": "exact",
    "testing.assert_raises": "exact",
    "testing.assert_raises_regex": "exact",
    "testing.assert_string_equal": "exact",
    "testing.assert_warns": "exact",
    "testing.break_cycles": "exact",
    "testing.build_err_msg": "exact",
    "testing.check_support_sve": "exact",
    "testing.clear_and_catch_warnings": "exact",
    "testing.decorate_methods": "exact",
    "testing.jiffies": "exact",
    "testing.measure": "exact",
    "testing.memusage": "exact",
    "testing.print_assert_equal": "exact",
    "testing.run_threaded": "exact",
    "testing.rundocs": "exact",
    "testing.runstring": "exact",
    "testing.suppress_warnings": "exact",
    "testing.tempdir": "exact",
    "testing.temppath": "exact",
    "testing.test": "exact",
    "testing.verbose": "exact",
}

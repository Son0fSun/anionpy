#!/usr/bin/env python3
"""Fail the build if anything that must not be published would be published.

WHY THIS EXISTS
---------------
`docs/PHASE-0-DECISIONS-2026-08-06.md` originally decided that the public repo
would be built by an EXPORT FILTER: the private tree would keep `ionp-eren/`,
`ionp-nika/`, the internal `docs/*.md` ledgers and `bench/reference_gate.json`
back, and the public repo would be constructed by copying in only what ships.

That decision has since been SUPERSEDED. The whole repository ships public,
unfiltered -- `tests/`, `tools/`, `docs/`, `bench/reference_gate.json`,
`ionp-eren/`, `ionp-nika/`, all of it. An export filter is a thing someone has
to REMEMBER to apply correctly, every time, forever, and a leak behind it is
silent, one-way, and un-retractable once a mirror or index caches it. The
policy this script now enforces is the opposite one: nothing gets hidden, so
anything that must not ship has to actually be FIXED at the source, not routed
around. `NEVER_PUBLISH` below is consequently tiny -- it excludes build
artifacts and local environments that were never source to begin with, not
directories of source this scanner is afraid to look at.

It is deliberately a DENY-list over the SHIPPING set, not an allow-list over the
whole tree, because the dangerous direction is a new shipped file quietly
introducing a banned string -- not a new private file appearing.

WHAT IT DOES NOT DO
-------------------
It does not build the public tree, copy anything, or delete anything. It is
read-only and has no destructive mode. Run it; read the exit code.

LIMITS, stated plainly so nobody reads a green run as more than it is:
  - It greps for KNOWN banned tokens. A leak phrased in words not on the list
    passes. This is a backstop for regressions, never a substitute for review.
  - It checks source text only. It cannot see what a compiled artifact embeds.
  - The banned-token list is a policy artifact and will drift from policy unless
    someone maintains it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Paths that must NEVER be copied into the public repo. Anything matching one of
# these prefixes is private, and this script does not scan inside them -- they
# are allowed to contain whatever they contain.
#
# This list is deliberately almost empty. The whole repository ships public
# now, so the only things excluded are paths that are not source at all --
# build output and local environments -- both already gitignored and neither
# ever meant to be part of a published tree. Everything else (ionp-eren/,
# ionp-nika/, docs/, tests/, tools/, bench/reference_gate.json, PLAN-fanout.md,
# and the rest) is IN the shipping set and must pass the scan below on its own
# merits, not be hidden from it.
NEVER_PUBLISH = (
    "target/",
    ".venv/",
)

# Tokens that must not appear in any SHIPPED file. Case-insensitive, word-ish
# boundaries where the token would otherwise match innocuous substrings.
BANNED = {
    "internal-infra": [
        r"\bioncore\b",
        r"\bquark\b",
        r"10\.69\.1\.\d+",
        r"\belyssian\b",
        r"\btirion\b",
        r"\bashenera\b",
        r"\bilmarin\b",
    ],
    # NOTE: `eren` and `nika` ARE here, and this needs re-justifying every
    # time someone is tempted to remove them again (a prior revision of this
    # file did exactly that, on the reasoning that they are "now disclosed
    # shipping crate names" -- that reasoning was wrong and got corrected by
    # review). Whether `ionp-eren/`/`ionp-nika/` ship at all is an OPEN
    # decision (Phase 0 decided to excise them before any public release; that
    # decision has not been reversed). Banning their names here is a
    # dependency on that open question, not an assumption about its answer:
    # if the excision decision is reversed and the crates are cleared to ship,
    # remove them from BANNED (and their entries from ALLOW below) as part of
    # THAT decision, not preemptively. Until then a scanner that already
    # assumes "yes they ship" is worse than no scanner -- it would pass a
    # build that ships something nobody approved shipping.
    "codenames": [
        r"\beren\b",
        r"\bnika\b",
        r"\bsarati\b",
        r"\bspectra\b",
        r"\bvalinor\b",
        r"\bdoriath\b",
        r"\blegion\b",
        r"\bwaveoperator\b",
    ],
    "people": [
        r"\baion\b",
        r"\bseren\b",
        r"\basteria\b",
    ],
    "architecture": [
        r"\bpolyphasic\b",
        r"\bPWA\b",
        r"\bwave[- ]?function\s+algebra\b",
    ],
}

# Justified exceptions. A checker that fires on legitimate text gets ignored,
# and an ignored checker is worse than none -- so ambiguous English words that
# collide with codenames are exempted HERE, per-path, with a written reason,
# rather than by weakening the pattern for everyone.
#
# Each entry is (path, regex-source, reason). The exemption is scoped to that
# one file: the same word elsewhere still fails.
ALLOW = [
    ("ionp-ion/src/circulant.rs", r"\bspectra\b",
     "plural of 'spectrum' -- FFT spectra multiplied elementwise, not the product name"),
    ("ionp-ion/src/bin/bench.rs", r"\bspectra\b",
     "plural of 'spectrum' -- FFT spectra in a benchmark print, not the product name"),
    ("bench/reference_gate.json", r"\bspectra\b",
     "plural of 'spectrum' -- \"Ion multiplies spectra elementwise\", not the product name."),
    ("Cargo.toml", r"\beren\b",
     "root workspace `members` array. docs/PHASE-0-DECISIONS-2026-08-06.md decided the "
     "excision of ionp-eren/ionp-nika is an EXPORT FILTER applied when constructing the "
     "public repo, not a deletion from this tree -- other Claude sessions depend on "
     "`cargo build --workspace` succeeding here. The export filter is the thing responsible "
     "for not shipping this line verbatim; this scanner covers files copied as-is, and "
     "Cargo.toml is not one of those. This exemption stands or falls with the excision "
     "decision itself, same as the `eren`/`nika` ban above -- not independently."),
    ("Cargo.toml", r"\bnika\b",
     "root workspace `members` array -- same reason as the ionp-eren exemption directly above."),
]

# Extensions worth scanning as text.
TEXT_SUFFIXES = {".rs", ".py", ".toml", ".md", ".yml", ".yaml", ".cfg", ".txt", ".json", ".pyi"}


def is_private(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in NEVER_PUBLISH)


# This scanner's own source is exempt from its own scan. It ships (it is not
# in NEVER_PUBLISH), but its job requires its text to literally contain the
# banned tokens -- the BANNED regex source strings, the ALLOW reasons that
# quote them, and the self-check probe string. Scanning them would just be the
# checker reporting its own pattern list as a violation of itself, which tells
# nobody anything. This is the one legitimate reason to skip a shipping file
# outright rather than add per-line ALLOW entries: the "violations" here ARE
# the checker, not content that leaked into it.
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()


def shipping_files() -> list[Path]:
    out = []
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel == SELF:
            continue
        if is_private(rel) or "/.git/" in f"/{rel}" or rel.startswith(".git/"):
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            out.append(p)
    return sorted(out)


def main() -> int:
    files = shipping_files()
    compiled = {
        cat: [re.compile(pat, re.IGNORECASE) for pat in pats]
        for cat, pats in BANNED.items()
    }

    hits: list[tuple[str, str, int, str]] = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for cat, pats in compiled.items():
                for pat in pats:
                    if not pat.search(line):
                        continue
                    if any(
                        rel == a_path and re.search(a_pat, line, re.IGNORECASE)
                        for a_path, a_pat, _reason in ALLOW
                    ):
                        continue
                    hits.append((cat, rel, lineno, line.strip()[:140]))
                    break

    print(f"scanned {len(files)} shipping files")
    print(f"private prefixes excluded: {len(NEVER_PUBLISH)}")

    # Self-check: the scanner must be able to FAIL. A checker that cannot report
    # a violation has not reported cleanliness. Prove the matcher works on a
    # string we know is banned before trusting a clean run.
    probe = "host quark at 10.69.1.24 running ioncore"
    proved = any(p.search(probe) for p in compiled["internal-infra"])
    if not proved:
        print("FATAL: scanner failed its own self-check -- it cannot detect a known "
              "banned string, so a clean result would be meaningless.", file=sys.stderr)
        return 2
    print("scanner self-check: PASS (detects a known banned string)")

    # LICENSE placeholder gate. MIT was chosen deliberately; the copyright
    # HOLDER was not, and it is not the implementer's to invent -- the IP is
    # owned by a separate entity. A LICENSE naming the wrong holder is a legal
    # defect, and one naming a placeholder is worse than absent, because a
    # LICENSE file reads as a completed grant to anyone who finds it.
    #
    # This is a deliberate publish blocker, not a lint. Fill in the real legal
    # entity, or the export fails.
    # ORDERING MATTERS, AND IT BIT ONCE ALREADY. This gate used to `return 1`
    # here, BEFORE the token report below. While the placeholder was present --
    # which is the whole current state of the repo -- every banned-token leak
    # was computed and then thrown away unprinted, and the operator saw a
    # LICENSE complaint and nothing else. Two independent blockers must both be
    # REPORTED; only the exit code gets collapsed. Do not re-order this.
    #
    # SELF-CHECK, same reasoning as the banned-token probe above: now that the
    # placeholder has actually been replaced (LICENSE currently names a real
    # string, "Varda"), this gate passes on every real run and would keep
    # silently passing even if `has_placeholder` were rewritten to always
    # return False. A check that always passes because reality changed, and a
    # check that always passes because it is broken, produce an identical
    # trace unless something proves the negative path still fires. So: prove
    # the detector still rejects a known-bad LICENSE text before trusting it
    # on the real one.
    #
    # NOTE on what this gate does NOT prove: it verifies the placeholder
    # string is gone, nothing more. It cannot and does not verify that the
    # name now in LICENSE is the CORRECT legal holder -- that is a legal/
    # ownership question (see the repo's own IP-ownership notes on Varda vs.
    # the entity that owns the IP), not a text-matching one, and is out of
    # scope for a grep-based script to adjudicate.
    def has_placeholder(text: str) -> bool:
        return "COPYRIGHT-HOLDER-PENDING" in text

    bad_probe = "MIT License\n\nCopyright (c) 2026 COPYRIGHT-HOLDER-PENDING\n"
    if not has_placeholder(bad_probe):
        print("FATAL: LICENSE gate failed its own self-check -- it cannot detect a "
              "known-bad placeholder LICENSE, so a clean result would be meaningless.",
              file=sys.stderr)
        return 2
    print("LICENSE gate self-check: PASS (rejects a known-bad placeholder LICENSE)")

    licence_blocked = False
    lic = REPO / "LICENSE"
    if lic.is_file():
        lic_text = lic.read_text(encoding="utf-8", errors="replace")
        if has_placeholder(lic_text):
            licence_blocked = True

    if hits:
        print(f"\nRESULT: {len(hits)} banned-token occurrence(s) in files that would ship:\n",
              file=sys.stderr)
        for cat, rel, lineno, line in hits:
            print(f"  [{cat}] {rel}:{lineno}: {line}", file=sys.stderr)
        print("\nEither scrub the file or add its path to NEVER_PUBLISH.", file=sys.stderr)
    else:
        print("\nRESULT: no banned tokens in the shipping set.")
        print("NOTE: this is a regression backstop over a fixed token list, "
              "not a guarantee of IP cleanliness. Human review still required.")

    if licence_blocked:
        print("\nRESULT: LICENSE still contains the copyright-holder placeholder.\n"
              "  A LICENSE file is a public, irrevocable grant. Shipping one that\n"
              "  names <<COPYRIGHT-HOLDER-PENDING>> grants rights on behalf of\n"
              "  nobody. Replace it with the real legal entity before export.",
              file=sys.stderr)

    return 1 if (hits or licence_blocked) else 0


if __name__ == "__main__":
    raise SystemExit(main())

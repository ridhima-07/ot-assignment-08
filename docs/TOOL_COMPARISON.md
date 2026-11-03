# AI tool comparison (Section 8) — template, to be completed by the student

Only Claude Opus 5.5 was used to generate this repository, so the comparison below must be run by the student
with a second tool. Use **one fixed task** with the identical prompt in both tools so the comparison is fair.

## Suggested task (small, checkable in minutes)

> Write a Python function `licm(code)` for a three-address IR given as a list of tuples
> (`("bin", op, dst, a, b)`, `("const", dst, v)`, `("label", L)`, `("jmp", L)`, `("jz", a, L)`, ...)
> that hoists loop-invariant computations of single-definition temporaries `tN` out of natural loops into the
> loop preheader. It must not hoist division or modulo by a non-constant. Include a test.

Check each answer by dropping it into `trackF/passes.py` in place of `licm` and running
`python3 trackF/fuzz.py 300` (0 mismatches required) and `python3 trackF/run_trackF.py`.

## Results

| Criterion | Claude Opus 5.5 | Second tool (name + version) |
|---|---|---|
| Passes the 300-program differential fuzz? | yes (the version in this repo, 1000 programs, 0 mismatches) | |
| Hallucinations (APIs/facts that do not exist) | e.g. assumed textbook xv6 `copyout` signature (Failure log #10) | |
| Prompt iterations to a correct answer | | |
| Wall-clock time to a correct answer | | |
| Notable bug | | |

## Observations to write up (2–3 sentences each)
* Which tool asked for missing information instead of inventing it?
* Which produced code that compiled but was wrong, and how was it caught?

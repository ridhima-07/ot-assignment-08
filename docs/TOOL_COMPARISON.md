# AI tool comparison (Section 8)

**Tools compared:** Claude Opus 5.5 (main tool for this repository) vs **Claude Haiku 5.5**, run as a separate agent
on 2026-11-03 with no access to the repository — only `minic.py` (to compile/run test programs) and the prompt below.

> **Caveat.** Both are Anthropic models. They differ in size, speed and cost, so this is a real model comparison, but a
> comparison with a different vendor's tool (ChatGPT, Gemini, Copilot…) would be stronger. Adding one is a good
> extension; `grade.py` makes it a 5-minute job: save the tool's answer as `docs/tool_comparison/<tool>/licm_solution.py`
> and run `python3 docs/tool_comparison/grade.py docs/tool_comparison/<tool>/licm_solution.py`.

## Task (identical prompt)
> Write a Python function `licm(code)` for a three-address IR given as a list of tuples (`("bin", op, dst, a, b)`,
> `("const", dst, v)`, `("mov", dst, src)`, `("print", a)`, `("label", L)`, `("jmp", L)`, `("jz", a, L)`, ...) that hoists
> loop-invariant computations of single-definition temporaries `tN` out of natural loops into the loop preheader. It must
> not hoist division or modulo by a non-constant. Include a test.

## Grading (same for both): `docs/tool_comparison/grade.py`
The candidate replaces the LICM pass inside the MiniC compiler; then 300 random programs × 3 pass pipelines are
differentially fuzzed, and the 5 Track F benchmarks are checked for output equality and executed instructions.
Raw output: `docs/tool_comparison/haiku_grade.json`; Haiku's code and its own tests: `docs/tool_comparison/haiku/`.

## Results

| Criterion | Claude Opus 5.5 (`trackF/passes.py::licm`) | Claude Haiku 5.5 (`haiku/licm_solution.py`) |
|---|---|---|
| Differential fuzz, 300 programs × 3 pipelines | 0 mismatches | 0 mismatches |
| Benchmarks: output identical | 5/5 | 5/5 |
| Executed instructions, **LICM alone**: primes | 60,931 → 50,636 | 60,931 → **49,445** |
| Executed instructions, **LICM alone**: matmul | 189,869 → 143,091 | 189,869 → **140,659** |
| Executed instructions, LICM alone: other 3 programs | identical to Haiku | identical to Opus |
| Executed instructions, full pipeline | identical (constant propagation hides the difference) | identical |
| Header entered by an outside jump | skips that loop (conservative) | creates a new preheader label and redirects the jump (more general) |
| Hoists `%`/`/` by a constant | only if the divisor is already an integer literal (i.e. after constant propagation) | also when the divisor is a temporary holding a non-zero constant, so it works without constant propagation |
| Size | ~35 lines inside the pass module | 210 lines + 171 lines of tests (8 tests incl. its own 400-program fuzz) |
| Prompt iterations to a correct answer | 1 for this pass (verified by the 1,000-program fuzz); other Opus code in this repo needed fixes (failure log) | 1 (its own tests passed first time; confirmed by our grader) |
| Time | part of a long session (not separately timed) | ≈ 2 min 14 s, 4 tool calls |
| Hallucinations observed | elsewhere: assumed the textbook xv6 `copyout` signature (failure #10) | none on this task |
| Assumptions stated unprompted | in code comments | yes, listed 8 explicitly (e.g. user variables never named `tN`) |

## Observations
* **Correctness was equal; quality was not.** The smaller model produced the *stronger* optimisation: it hoists out of
  nested loops more aggressively and handles constant divisors, saving 2–3% more instructions on the loop-heavy
  programs. The larger model's version was simpler and more conservative. A bigger model is not automatically better on
  a narrow, well-specified task.
* **Self-reported tests are not enough.** Haiku said "8/8 tests pass"; that claim was only accepted after the
  independent grader (fuzzing against the real compiler) agreed. The same rule caught 15 Opus bugs (`FAILURE_LOG.md`).
* **Stating assumptions** (Haiku listed eight, e.g. that source variables never look like temporaries) is what
  Section 8's prompt asks for ("ask me instead of inventing it"); neither model asked a question, both stated assumptions.
* **Follow-up:** Haiku's LICM could replace the repository's version; that is left as a deliberate exercise
  (`docs/VIVA_NOTES.md`, Track F) so the change can be made and explained by the student.

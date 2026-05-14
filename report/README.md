# Report

Springer LNCS template for the term paper (6–8 pages).

## Build

1. Download `llncs.cls` and `splncs04.bst` from the [Springer LNCS guidelines page](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines) and place them in this directory. They are not redistributed here for licensing reasons.
2. Build with:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or with `latexmk -pdf main`.

## Status

The `.tex` file already contains all eight required sections (Introduction,
Related Work, Methodology, Experimental Setup, Results, Error Analysis,
Discussion, Conclusion). Result tables have `\dots` placeholders — fill them
in from `results/summary.md` after running `scripts/run_eval.sh` and
`scripts/run_ablations.sh`.

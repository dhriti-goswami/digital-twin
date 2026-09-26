# Updated paper — figures replaced in-place (v2, hardened)

`main.tex` is your original paper with Figures 2-8 replaced directly in
place with native PGFPlots code reading the repository CSVs in `data/`.
Nothing else in the paper's text, tables, captions, or labels was changed.

## What was fixed in this version
A prior version had two real rendering bugs, now fixed:
1. **Fig. 8 (permutation importance) was silently dropping a bar.** It used
   a multi-line list of symbolic y-axis names; a stray leading space before
   one entry (`glucose_mean_1h`) could fail to match the data on some
   PGFPlots versions, leaving that bar blank. Rebuilt using plain numeric
   y-positions with `yticklabels` instead of string-matched symbolic
   coordinates — this can't silently fail to match.
2. **Legends were missing on the second panel of Fig. 2 and Fig. 7.** Only
   the first panel (MAE / TIR) had a `\legend{...}` call; the second panel
   (RMSE / TBR) plotted the same two series with no legend. Both panels now
   have their own legend.

Also bumped legend font from `\tiny` to `\scriptsize` for readability, and
re-verified every figure at 200 DPI on the actual two-column IEEEtran output
(not just an isolated test file) to confirm nothing clips or overlaps.

## Files
- `main.tex` — the full paper, ready to compile.
- `main.pdf` — already compiled (16 pages), so you can check it immediately.
- `data/` — the 12 CSVs the figures read from. Keep this folder next to
  `main.tex` — if it's moved or renamed, the figures will fail to find their
  data (this is the most common cause of "missing values" with this setup).

## To compile it yourself
Requires `pgfplots`, `pgfplotstable`, and IEEEtran
(`apt install texlive-publishers` on Debian/Ubuntu covers IEEEtran). Then:
```
pdflatex main.tex
pdflatex main.tex   # second pass, resolves cross-references
```
Compiled twice in my environment with zero errors and zero warnings about
missing figures/data.

## Still true from before
- Fig. 6 (hypoglycemia alarm trade-off) has no source CSV anywhere in the
  repository — its six values come from `docs/RESULTS.md`, flagged in a
  code comment directly above that figure in `main.tex`.
- Fig. 8's permutation-importance panel shows the CSV's true top-9 features
  in order (the original PNG had dropped two real top-9 features and
  substituted two lower-ranked ones out of order) — see MISMATCH_REPORT.md.

# Merge notes (v4)
Base: original v3 main.tex. Ours: digital_twin_v4_final_dg_before_merge.zip. Theirs: GitHub dhriti-goswami/digital-twin @ 6a7ad37.
3-way merge; every change from both sides kept. Two overlaps, resolved as:
1. Preregistration/IG area: Samyak's new Reproducibility section + Dhriti's edited IG line.
2. Limitations para 1: Samyak's paragraph + Dhriti's last sentence ("physics-guided forecaster rather than a digital twin or treatment simulator").
Extra edits made during merge (formatting only, no content/number changes):
- Restored the missing \subsection{Background} heading (deleted accidentally when pasting).
- Equation lead-ins normalized to ", as given in Eq.~(x):" (10 places) incl. "Attention pooling, Eq.~(14), gives".
- References renumbered by first appearance; consecutive numbers printed as [\hyperref..a-..b] ranges, others via \cite. See REFERENCE_MAP.md.
- ref4 dropped from the bibliography because nothing cites it any more.
Compiles: 19 pages, no errors, no undefined references.

# v4 fixes (after merge)
- Physics Ablations: contradicting sentence rewritten ("Beyond this modest accuracy gain, ...").
- Abstract Conclusion replaced (old clinical-translation claim removed); abstract Methods "combines  ... to" -> "couples ... to".
- Roadmap sentence matches what exists; 5 notation tables moved to supplementary.tex; PRED-EGA row no longer says "not implemented".
- "Digital twin" removed for our model in Samyak's sections (7 places); reviewer language removed; "three" -> "four" paradigms.
- PRED-EGA sentence removed from Future Work; IG paragraph given its own heading "Feature Attribution Method".
- "missing 43%" -> "missing 45% of hypoglycemic readings (pointwise sensitivity 0.551; Table IX)" (45 = 1 - 0.551).
- Captions explaining mismatching values: Tables IV, V, VI, VII, IX. Run attribution based on data/ CSV labels -> Samyak to confirm.
- No new decimal numbers introduced (checked automatically). References: no renumbering needed after the table move.

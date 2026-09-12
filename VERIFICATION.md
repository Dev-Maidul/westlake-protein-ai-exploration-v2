# Independent Re-Verification

Every result in [PROGRESS.md](./PROGRESS.md) was re-run from scratch on **2026-09-12** to confirm it
wasn't a one-off fluke, with exact commands, timing, and environment recorded below.

| Repo | Command re-run | Environment | Wall time |
|---|---|---|---|
| SaProt | `python demo_mutation_effect.py R175H example/p53_AF-P04637.pdb` | `saprot-env` venv, **MPS** | 13.7s |
| ProTrek (seq→text) | `python demo_protrek_search.py` | `saprot-env` venv, **MPS** | 8.3s |
| ProTrek (struct→text) | `python demo_protrek_structure_search.py` | `saprot-env` venv, **MPS** | 7.6s |
| Pinal | `PINAL_MODEL_ROOT=weights python demo_pinal_design.py "Actin." 2` | `pinal-env` venv, **CPU** | 20m 5s |

---

## SaProt — p53 R175H mutation score

```
Wild-type combined token at position 175: 'Ra'
Mutant combined token at position 175:    'Ha'

log P(wild-type token)  = -3.2464
log P(mutant token)     = -4.6403
Mutation score (mt - wt) = -1.3939
=> Model predicts this mutation is UNFAVORABLE (worse than wild-type).
```

**Result:** identical to the first run — **score −1.3939**. This is expected: no randomness is involved
in this scoring method, so it's fully deterministic run to run.

---

## ProTrek — sequence→text (human insulin)

```
Similarity scores against candidate function descriptions:
  +12.844  |  A hormone that regulates blood glucose levels
   +4.293  |  An enzyme that digests starch into sugars
   +2.783  |  A structural protein found in hair and nails
   -0.236  |  A photosynthesis-related protein found in plants
   -2.599  |  A protein involved in DNA repair
```

**Result:** identical to the first run, all five scores matching exactly (cosine similarity between
fixed embeddings — deterministic, no sampling).

## ProTrek — structure→text (p53, no sequence used)

```
Similarity scores (structure-only input, no amino-acid sequence used):
   +8.769  |  A tumor suppressor protein that prevents cancer by regulating the cell cycle
   +6.162  |  A hormone that regulates blood glucose levels
   +5.857  |  A structural protein found in hair and nails
   -2.117  |  An enzyme that digests starch into sugars
   -3.882  |  A photosynthesis-related protein found in plants
```

**Result:** identical to the first run.

---

## Pinal — text-guided design ("Actin.")

```
Run 1 (original)     score -12.861  MCDEETAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKD...
Run 2 (re-verified)  score -13.058  MCDEEETAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKD...
```

**Result: not byte-identical — flagging this explicitly rather than glossing over it.** The design script
uses `saprot_sample_type="multinomial"`, i.e. it *samples* each amino acid from a probability distribution
rather than always taking the top prediction, so some variation between runs is expected by design (not a
bug). What stayed consistent across both independent runs: the same recognizable N-terminal region
matching real human beta-actin, and a similar confidence score (~ -13). Running with
`saprot_sample_type="argmax"` instead would give a fully reproducible (greedy) output at the cost of
sequence diversity.

---

## Evolla — hardware/quantization feasibility check

Investigated two possible paths to running Evolla-10B on more accessible hardware:

**1. Kaggle's free P100 (16GB VRAM)**

Checked actual file sizes on `westlake-repl/Evolla-10B-hf` via the HuggingFace API:

| Precision | Size |
|---|---|
| F32 (only format the checkpoint ships in) | **41.57 GB** |
| FP16 (would require re-casting) | ~20.78 GB |
| INT8 | ~10.39 GB |
| INT4 (NF4) | ~5.2 GB |

Even INT4 would technically fit in 16GB VRAM, but there is no pre-quantized checkpoint — quantizing
on-the-fly still requires downloading the full 41.57GB F32 checkpoint first, which exceeds Kaggle's
typical working-directory disk quota. P100 is also Pascal-generation (compute capability 6.0); bitsandbytes'
efficient 4-bit/8-bit kernels primarily target Turing+ (7.5+), so even if it fit, performance is uncertain
(not benchmarked directly — flagging as a reasoned estimate, not a verified fact).

**2. Quantized/GGUF community versions**

Searched the HuggingFace Hub directly (`Evolla GGUF`, `Evolla 4bit`, `Evolla quantized`) — no results.
Only the five official checkpoints exist (`Evolla-10B`, `-10B-hf`, `-10B-DPO`, `-10B-DPO-hf`, `-80B`), all
full-precision.

**Conclusion:** not running Evolla locally or on Kaggle isn't a matter of trying harder — it's a real
infrastructure gap (no lightweight checkpoint exists yet for this model family).

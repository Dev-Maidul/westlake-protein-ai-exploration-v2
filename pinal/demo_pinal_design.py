"""
Text-guided de novo protein design with Pinal (local, no cloud GPU).

Pipeline: text description -> T2struc-1.2B (text -> structure tokens)
                             -> SaProt-T (structure tokens -> amino acid sequence)

Uses the smallest available checkpoints (T2struc-1.2B + SaProt-T, ~7.5GB total)
instead of the full 16B server version.
"""
import sys
import torch
import utils.design_utils as design_utils

# Note: this torch version's MPS backend does not support bfloat16 (which
# these models are loaded in), so we stay on CPU despite having Apple Silicon.
print(f"Using device: {design_utils.DEVICE}")

from utils.design_utils import load_pinal, PinalDesign

DESCRIPTION = sys.argv[1] if len(sys.argv) > 1 else "Actin."
NUM_DESIGNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print(f"\nLoading Pinal (T2struc-1.2B + SaProt-T) ... this may take a while and use significant RAM.")
t2struc, text_tokenizer, structure_tokenizer, saprot, saprot_text_tokenizer, saprot_tokenizer = load_pinal()

print(f"\nDesigning {NUM_DESIGNS} protein(s) for description: \"{DESCRIPTION}\"")
res = PinalDesign(
    desc=DESCRIPTION,
    num=NUM_DESIGNS,
    t2struc=t2struc,
    text_tokenizer=text_tokenizer,
    structure_tokenizer=structure_tokenizer,
    saprot=saprot,
    saprot_text_tokenizer=saprot_text_tokenizer,
    saprot_tokenizer=saprot_tokenizer,
    temperature=1.0,
    saprot_sample_type="multinomial",
    multinomial_temperature=0.3,
)

print("\nDesigned protein sequences (sorted by model confidence):")
for i, seq in enumerate(res, 1):
    print(f"\n[{i}] length={len(seq)}")
    print(seq)

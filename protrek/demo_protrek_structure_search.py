"""
Structure <-> text search with ProTrek: given only a protein's 3D structure
(no amino acid sequence used for this embedding), find which natural-language
function description matches best. This tests the third leg of the trimodal
model (structure-text alignment), complementing the sequence-text demo.
"""
import torch
import torchmetrics
import functools

torchmetrics.Accuracy = functools.partial(torchmetrics.Accuracy, task="multiclass", num_classes=2)

import utils.lr_scheduler as lr_scheduler_module

def _patched_constant_init(self, optimizer, last_epoch=-1, verbose=False, init_lr=0.):
    self.init_lr = init_lr
    lr_scheduler_module._LRScheduler.__init__(self, optimizer, last_epoch)

lr_scheduler_module.ConstantLRScheduler.__init__ = _patched_constant_init

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
    def _batch_encode_plus(self, texts, **kwargs):
        return self(texts, **kwargs)
    PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq

WEIGHTS_DIR = "weights/ProTrek_35M_UniRef50"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Extracting 3Di structure sequence from p53 AlphaFold structure ...")
aa_seq, struc_seq, combined_seq = get_struc_seq(
    "bin/foldseek", "example/p53_AF-P04637.pdb", ["A"], plddt_mask=False
)["A"]
print(f"Protein length: {len(aa_seq)}")
print(f"3Di structure sequence (first 60): {struc_seq[:60]}")

config = {
    "protein_config": f"{WEIGHTS_DIR}/esm2_t12_35M_UR50D",
    "text_config": f"{WEIGHTS_DIR}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": f"{WEIGHTS_DIR}/foldseek_t12_35M",
    "from_checkpoint": f"{WEIGHTS_DIR}/ProTrek_35M.pt",
}
print("Loading ProTrek_35M model ...")
model = ProTrekTrimodalModel(**config).eval().to(device)

candidate_texts = [
    "A tumor suppressor protein that prevents cancer by regulating the cell cycle",
    "A hormone that regulates blood glucose levels",
    "A structural protein found in hair and nails",
    "An enzyme that digests starch into sugars",
    "A photosynthesis-related protein found in plants",
]

with torch.no_grad():
    # lowercase 3Di string is what the structure encoder's tokenizer expects
    struc_emb = model.get_structure_repr([struc_seq.lower()])
    text_emb = model.get_text_repr(candidate_texts)
    scores = (struc_emb @ text_emb.T / model.temperature).squeeze(0)

print("\nSimilarity scores (structure-only input, no amino-acid sequence used):")
ranked = sorted(zip(candidate_texts, scores.tolist()), key=lambda x: -x[1])
for text, score in ranked:
    print(f"  {score:+7.3f}  |  {text}")

print(f"\nBest match: \"{ranked[0][0]}\"")

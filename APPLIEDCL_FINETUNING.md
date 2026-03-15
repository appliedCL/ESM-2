# AppliedCL: ESM-2 fine-tuning for RIMNet

This fork supplies the credited Meta ESM implementation used by AppliedCL's
RIMNet protein-interaction research. The original Meta architecture and
pretrained weights remain upstream work under the [MIT license](LICENSE).
AppliedCL's task-specific model and training implementation are in the
[RIMNet repository](https://github.com/appliedCL/RIMNet).

## What is fine-tuned

RIMNet uses `esm2_t33_650M_UR50D`: freeze the first 31 transformer blocks and
update the final two complete blocks plus an original chain-moment interaction
head. This is partial full-block fine-tuning, not LoRA. A matched frozen arm
updates only the same head. The head receives separate mean/standard-deviation
summaries for the two jointly encoded proteins. Evaluation averages both input
orders. The learning rates are 0.00002 for ESM-2 and 0.0003 for the head, with
AdamW, binary cross-entropy, and effective batch size 32.

The verified Mac warm-start made 16 optimizer updates (512 training examples)
per arm. It confirms that selected weights update, frozen weights remain
unchanged, and checkpoints reload correctly. It supplies no validation AUROC or
claim of improved generalization. Full training is a three-epoch development
experiment continued on Kaggle; the final test is excluded from the transfer.

## Implementation and Kaggle notebook

- [Portable ESM interface and task-specific head](https://github.com/appliedCL/RIMNet/blob/main/src/protbert_ppi/portable_esm.py)
- [Resumable training implementation](https://github.com/appliedCL/RIMNet/blob/main/src/protbert_ppi/finetuning.py)
- [Mac preparation and verification](https://github.com/appliedCL/RIMNet/blob/main/scripts/finetune_esm2.py)
- [Kaggle notebook](https://github.com/appliedCL/RIMNet/blob/main/notebooks/RIMNet_ESM2_finetuning_Kaggle.ipynb)
- [Full protocol and scientific limitations](https://github.com/appliedCL/RIMNet/blob/main/docs/ESM2_FINETUNING.md)

Use the RIMNet preparation tool with this checkout supplied via `--esm-source`.
It converts the locally cached, pinned Hugging Face ESM-2 weights, verifies
numerical agreement, and builds a local fair-esm dependency wheel carrying the
original license. The Kaggle transfer contains base weights, development data,
code wheels and warm-start checkpoints, all excluded from Git. No private
credentials or pretrained weights are committed here. See the full protocol
for exact commands and offline notebook setup.

The upstream Python model implementation is unchanged by this integration.
RIMNet's task-specific implementation is maintained in one repository; this
fork links to it rather than duplicating that training code. No PIFM code is
used in the RIMNet implementation.

## Required Kaggle input

The notebook alone does not contain the model, dataset or warm-start files.
From the RIMNet checkout, run `python3 scripts/package_kaggle_transfer.py` after
preparation and the verified warm-start. Upload the resulting
`runs/rimnet-kaggle-upload-v1.zip` as a **private Kaggle Dataset**, wait for
processing, then attach it through the notebook's **Input → Add Input**.
Kaggle unpacks ZIP uploads, preserving `transfer.json` and its sibling folders.
If multiple transfer manifests are attached, the updated notebook lists the
available `INPUT_ROOT` values. Use a Kaggle-mounted path, not a Mac path.
The upload archive and weights remain local and excluded from both Git repos.

Kaggle may additionally unpack the development CSV's inner gzip archive. The
current RIMNet notebook accepts either `.csv.gz` or the extracted `.csv`, checking
the original content checksum before training. Existing uploaded transfer v1 is
supported without another upload; use the updated notebook or the compatibility
block in the full RIMNet protocol. The pretrained weights, training configuration
and checkpoint identities stay unchanged.

The continuation notebook validates the saved update inside each checkpoint.
For the current run it requires step 1598 or later and rejects the original
step-16 transfer and the accidental step-40 restart. Set `RESUME_OVERRIDE` to
the prior saved Kaggle output checkpoint. For an intentional first run from the
Mac weights, explicitly set `ALLOW_WARM_START = True` in the first setup cell.
The T4 runtime now selects native FP16; BF16 is selected only on GPUs with native
support. No speedup estimate is claimed before measuring that runtime.

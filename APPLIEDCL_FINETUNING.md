# AppliedCL: ESM-2 fine-tuning for RIMNet

This fork supplies the credited Meta ESM implementation used by AppliedCL's
RIMNet protein-interaction research. Meta's architecture and pretrained weights
remain upstream work under the [MIT license](LICENSE). The task-specific model
and training workflow are original [RIMNet](https://github.com/appliedCL/RIMNet)
code; no PIFM source is used.

## One-cell Kaggle sessions

Use the [RIMNet Kaggle notebook](https://github.com/appliedCL/RIMNet/blob/main/notebooks/RIMNet_ESM2_finetuning_Kaggle.ipynb).
It has one complete execution cell: run it for up to eight hours, then run the
same cell again to continue. It rebuilds setup state, reads checkpoint metadata
and selects the furthest compatible saved update.

Attach the prepared transfer Dataset, enable a GPU, and enable **Session options
→ Persistence → Files only** before the session ends. Every optimizer update is
saved atomically; `previous.pt` retains a previous valid save. A progress journal
prevents large rollback. At each pause a downloadable `resume-bundle.zip`
contains available checkpoints and run metadata. Keep saved outputs or the ZIP
as an additional backup. An `.ipynb` file does not embed the learned weights.
Background jobs need prior saved outputs attached as inputs; the editor's
persistence setting does not provide their working state.

The current minimum step is 48, the latest checkpoint actually confirmed
available. An earlier run reported step 1598, but its checkpoint has not been
recovered. The notebook does not claim those weights were restored. For an
intentional first run from the original Mac update-16 checkpoint, explicitly
set `allow_initial_warm_start = True` in the cell's settings. Leave it disabled
for normal continuation.

The runtime selects native FP16 on T4; BF16 is used only where natively
supported. No GPU speedup is claimed from CPU tests. The original scientific
training code and checkpoint identities stay unchanged, and the session runtime
is embedded so the existing input Dataset needs no re-upload. Kaggle may unpack
the inner gzip CSV; exact content hashes are verified in either representation.

## Scientific implementation

RIMNet freezes the first 31 blocks of `esm2_t33_650M_UR50D` and trains the final
two full transformer blocks plus an original chain-moment interaction head.
The matched frozen control updates only the head. This is partial full-block
fine-tuning. Both proteins are jointly encoded; separate mean/std summaries
feed the symmetric interaction head, and evaluation averages input orders.
The fixed budget is three epochs on the original development split. Final-test
records are excluded from training and checkpoint selection.

- [Portable backbone and original interaction head](https://github.com/appliedCL/RIMNet/blob/main/src/protbert_ppi/portable_esm.py)
- [Pinned scientific training loop](https://github.com/appliedCL/RIMNet/blob/main/src/protbert_ppi/finetuning.py)
- [Checkpoint/session wrapper](https://github.com/appliedCL/RIMNet/blob/main/src/protbert_ppi/notebook_session.py)
- [Full protocol, setup and limitations](https://github.com/appliedCL/RIMNet/blob/main/docs/ESM2_FINETUNING.md)

Use the RIMNet preparation tool with this source checkout supplied via
`--esm-source`. It converts locally cached pinned weights, verifies numerical
agreement and builds a local fair-esm dependency wheel carrying the original
license. Upload archives, datasets, model weights, optimizer checkpoints and
logs remain excluded from Git. The upstream Python model source is unchanged.
Code updates can be saved privately through the official Kaggle CLI's no-run
option without starting a training job. Editor persistence is a separate
Kaggle setting. Session recovery tests establish software correctness; no
AUROC improvement or achievement of >0.90 is asserted by those tests.

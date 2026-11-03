# Supervised training readiness

The two-epoch CPU smoke passed on 3 November 2026. This verifies data preparation,
training mechanics and checkpoint reload. Full 675-epoch reproduction, JEPA
benefit and biological feasibility/calibration are not established.

## Reference and configuration

Upstream: https://github.com/diogo-f-oliveira/DEB_ML_Init

The acquired commit is `be4494174931406f5b2a9beddd7da918df0f19dd`, identical to the
previously inspected reference. Local ref `baseline/upstream-be449417` preserves it.
Dataset originals remain unchanged in `data/processed/init_net`; observed counts
are 1,647 training, 353 validation and 353 test species (2,353 total).

The upstream trainer example cannot construct its empty-shared-layer setting.
The retained `logQ_81327e-01_MLP.pth` checkpoint instead has two shared layers of
width 128, dropout 0.24, ten linear heads and an input skip connection. Our
configuration reconstructs that architecture; strict state loading and exact
forward comparison against the trusted source checkpoint pass. No architecture
patch was made to upstream source. The optimizer settings come from the upstream
trainer example; this combination is evidenced for setup, not proven to be the
complete author-used training configuration. See `baseline-evidence.json`.

The runner calls the original data preparation, DEBNetHC, MaskedMSELoss through
the original training function, AdamW and validation evaluators. Output targets
are transformed fitted DEB estimates, with original eligibility rules. Test CSV
integrity is checked but test prediction performance is never evaluated by this
runner. Upstream validation's reported DEB Loss is its prediction diagnostic,
not a new mechanistic calibration run.

## Environment

Use Conda environment `deb-jepa`, Python 3.10.22 on this Apple Silicon host.
`environment.yml` uses conda-forge and `requirements-lock.txt` records installed
packages, including upstream Torch 2.5.1. The trainer imports Ray's train/tune
modules, so PyArrow 17.0.0 and tensorboardX are additional runtime dependencies.
No Ray cluster or tuning service is started. Recreating this environment on other
platforms may require a separately tested lock.

From the checkout root, this creates the named environment from its specification:

```bash
/opt/miniconda3/bin/conda env create -f environment.yml
```

Do not recreate an existing environment blindly. The current environment has
passed dependency consistency and runtime checks.

## Verified commands

These commands assume the checkout root as the working directory. Use fresh
output directories; the runner refuses to overwrite an existing run.

Run data/model preparation without optimizer updates:

```bash
/opt/miniconda3/envs/deb-jepa/bin/python -m project_code.train.run_baseline --mode dry-run --output results/readiness/dry-run-next
```

Run the bounded supervised smoke with seed 42, two epochs, CPU, validation only,
a 600-second deadline and a 100 MB output ceiling:

```bash
/opt/miniconda3/envs/deb-jepa/bin/python -m project_code.train.run_baseline --mode smoke --output results/readiness/smoke-next
```

Run the focused checks for data integrity, preprocessing, loss and source-model
architecture compatibility:

```bash
/opt/miniconda3/envs/deb-jepa/bin/python -m unittest discover -s tests -v
```

The smoke performed 3,294 optimizer steps; all checked gradients and metrics were
finite, weights changed, and reloaded validation predictions were bit-identical.
The observed smoke took approximately 5.73 seconds and wrote 174,153 bytes. These
are local mechanical results, not performance comparisons. Detailed source hashes,
package versions, data manifest and checkpoint are retained in ignored
`results/readiness/smoke-20261103`. A compact result is versioned in this directory.
New checkpoints store tensors and primitive metadata and load with
`weights_only=True`. Only the explicitly trusted upstream checkpoint is loaded as
a whole-object pickle in the architecture compatibility test.

## Proposed full-training invocation — not executed

After the baseline's training provenance and run budget are agreed, the following
programmatic invocation uses the original trainer and the separately recorded
675-epoch configuration. It performs validation diagnostics, saves a model and
does not request test scoring. It has not been verified at full duration and is
outside the accepted smoke authorization. The bounded runner deliberately offers
no unrestricted training mode.

```bash
/opt/miniconda3/envs/deb-jepa/bin/python - <<'PYRUN'
import json
from pathlib import Path
import torch
from project_code.algorithms.neural_networks import DEBNetHC
from project_code.train.train_pytorch_model import train_neural_network
from project_code.utils.models import save_pytorch_model
cfg = json.loads(Path('configs/hc_initnet.json').read_text())
out = Path('results/readiness/full-baseline')
out.mkdir(parents=True, exist_ok=False)
model, metrics = train_neural_network(cfg['training'], DEBNetHC, 'init_net',
                                    device=torch.device('cpu'), seed=cfg['seed'])
save_pytorch_model(model, str(out), 'hc_initnet')
metrics.to_json(out / 'validation-metrics.json')
PYRUN
```

This is a proposed launch example, not a replacement for a full experiment record.
No claim of published-result reproduction should be made from this setup alone.

## Review workflow and remaining science

Work on small `codex/` branches in your fork. Diogo can review proposed changes
there once João requests collaboration; pushing a branch does not imply a review
has occurred. No PR, invitation, reviewer request or message was created.

Tese remains the owner of research planning and learning materials. The sibling
`deb-jepa` checkout owns implementation, runtime and experiment configurations.
The current setup leaves the JEPA encoder boundary, masks, anti-collapse mechanism,
matched controls and primary biological endpoint open. Temperature/mu_V semantics,
MATLAB/DEBtool availability, calibration ownership and workload must be settled
before calibration claims or sweeps. A lower supervised loss alone is insufficient.

Local review checked split isolation, source reuse, CPU/deadline/output bounds,
serialization, provenance and scope. No independent agent review ran. Maintained
source/data/runtime are protected. Local dry-run and smoke outputs are retained
readiness evidence, not disposable QA; no separate temporary collection was created.

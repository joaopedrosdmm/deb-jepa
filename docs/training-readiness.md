# Supervised training readiness

The two-epoch CPU smoke and the three-finding remediation passed on 3 November 2026. This verifies data preparation,
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
the upstream training loop, AdamW and validation evaluators. The loop now has
optional step/epoch accounting; its default model/metrics return contract and
optimization behavior are preserved. Output targets
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

Run the focused checks for data integrity, preprocessing, loss, source-model
compatibility, checkpoint metadata, dataset routing and optimizer accounting:

```bash
/opt/miniconda3/envs/deb-jepa/bin/python -m unittest discover -s tests -v
```

The smoke performed 3,294 optimizer steps; all checked gradients and metrics were
finite, weights changed, and reloaded validation predictions were bit-identical.
The post-remediation smoke took approximately 4.29 seconds and wrote 175,238 bytes. These
are local mechanical results, not performance comparisons. Detailed source hashes,
package versions, data manifest and checkpoint are retained in ignored
`results/readiness/smoke-remediation-20261103`. The original
`smoke-20261103` evidence is retained as history. A compact result is versioned in this directory.
New checkpoints store tensors and primitive metadata and load with
`weights_only=True`. Reload reconstructs the scalers from saved column metadata
and statistics, and rejects mismatching model settings, preprocessing or column
order. Earlier readiness checkpoints lacking full metadata are rejected with a
regeneration instruction; they remain historical evidence. Only the explicitly
trusted upstream checkpoint is loaded as
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


## Review corrections — 3 November 2026

The subsequent local review found three P2 issues that did not invalidate the
original default smoke, but made its runner unsafe to extend unchanged. All
three are corrected in one local repair pass; no independent review is claimed.

| Finding | Correction | Evidence |
| --- | --- | --- |
| Reload ignored saved configuration/columns | Compare model settings and complete ordered column/transform metadata; construct model and scalers from checkpoint | Exact transformed and decoded predictions; changed clamp, architecture, preprocessing and order rejected |
| Preflight/training paths differed | Resolve `dataset` once and pass its parent/name to both upstream loaders and trainer; retain full path in manifest/report | Mocked alternative-location routing tests plus real original-data smoke |
| Update counts were inferred | Count actual optimizer steps and completed epochs; report selected checkpoint epoch separately | Batch-size-two, accumulated-gradient and early-stopping fixtures; real smoke 3,294 counted steps |

Verification: 21/21 unittest checks, `pip check`, whitespace check, zero-update
dry-run and the bounded two-epoch smoke passed. No test-performance scoring ran.
Post-fix state tensors and validation metrics match the original default smoke
exactly; all five source file hashes also match. Runtime source hashes in the
run report identify the repaired, then-uncommitted code; its `code_commit` is the
base HEAD at execution, not a claim that those edits were already committed.

The canonical Tese readiness plan retains the acceptance criteria, repair history,
remaining scientific choices and commit handoff. These corrections establish
mechanical readiness only. Full baseline reproduction still requires author-used
training provenance and an agreed run budget.

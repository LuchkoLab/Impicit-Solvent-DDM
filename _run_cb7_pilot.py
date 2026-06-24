"""Scratch driver: run the cb7 DDM workflow with the ALS pilot (Phase 4.5) ENABLED.

Step 5a smoke test. Sets workflow.adaptive_lambda=True + a short intermediate_args.pilot_nstlim so the
new adaptive_restraint_pilot runs: a short-MD pilot of the complex leg that drives the R-ADD restraint
scheduler to convergence and LOGS the converged schedule. Production Phases 5/6/7 are UNCHANGED (they
read the Phase-4 seed setups), so the cb7 free energy is the same as a flag-off run — the pilot is a
preview.

What to look for in the Toil log (grep '[ALS][pilot]'):
  - "[ALS][pilot] Phase 4.5 starting. pilot output dir: .../mdgb_pilot"   (R-DIR isolation)
  - per-iteration "[ALS][complex] inserting restraint window ..." / overlap logs (engine)
  - "[ALS][pilot] CONVERGED complex restraint schedule: N windows" with con/orient exponents
And confirm the production tree (mdgb/) is untouched by the pilot (pilot writes only to mdgb_pilot/).

Run on a cluster with AMBER + isddm_env (cb7 is CPU-only):
    PILOT_NSTLIM=50 SCRATCH=./cb7_pilot_run python _run_cb7_pilot.py

Untracked scratch file (mirrors _run_cb7_overlap.py). Routes outputs + jobstore to a LOCAL dir.
"""
import os

import yaml

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.implicit_ddm_workflow import ddm_workflow
from toil.common import Toil
from toil.job import Job

PILOT_NSTLIM = int(os.environ.get("PILOT_NSTLIM", "50"))   # short pilot MD length (production cb7 = 100)
SCRATCH = os.environ.get("SCRATCH", os.path.abspath("./cb7_pilot_run"))
os.makedirs(SCRATCH, exist_ok=True)

options = Job.Runner.getDefaultOptions(os.path.join(SCRATCH, "jobstore"))
options.logLevel = os.environ.get("TOIL_LOG", "INFO")      # INFO so the [ALS][pilot] logs show
options.clean = "always"
options.workDir = SCRATCH

with open("implicit_solvent_ddm/tests/input_files/config.yaml") as fh:
    cfg_dict = yaml.safe_load(fh)

config = Config.from_config(cfg_dict)

# --- enable the ALS pilot (Phase 4.5) ---
config.workflow.adaptive_lambda = True
config.intermediate_args.pilot_nstlim = PILOT_NSTLIM

# Route every output onto the local scratch fs (so this also works on a Mac sshfs mount).
config.system_settings.working_directory = SCRATCH
config.system_settings.cache_directory_output = SCRATCH
config.workflow.ignore_receptor_endstate = False

os.makedirs(config.system_settings.top_directory_path, exist_ok=True)

print(
    f"[driver] scratch={SCRATCH}  adaptive_lambda=True  pilot_nstlim={PILOT_NSTLIM}\n"
    f"[driver] production tree: {config.system_settings.top_directory_path}\n"
    f"[driver] pilot tree (expected): {config.system_settings.top_directory_path}_pilot\n"
    f"[driver] starting Toil workflow ...",
    flush=True,
)
with Toil(options) as toil:
    config.endstate_files.toil_import_parameters(toil=toil)
    config.intermediate_args.toil_import_user_mdin(toil=toil)
    config.inputs["min_mdin"] = str(
        toil.import_file("file://" + os.path.abspath("implicit_solvent_ddm/tests/input_files/min.mdin"))
    )
    toil.start(Job.wrapJobFn(ddm_workflow, config))

print("[driver] WORKFLOW DONE", flush=True)
prod = config.system_settings.top_directory_path
pilot = prod + "_pilot"
print(f"[driver] production dir exists: {os.path.isdir(prod)} ({prod})", flush=True)
print(f"[driver] pilot dir exists:      {os.path.isdir(pilot)} ({pilot})", flush=True)

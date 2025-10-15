# DDM Workflow Structural Flow Chart

## Overview
This document shows the structural flow of the Double Decoupling Method (DDM) workflow as implemented in `implicit_ddm_workflow.py`.

## Simplified Workflow Flow

```
                    DDM WORKFLOW START
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │        PHASE 1: SETUP               │
    │   setup_workflow_components()       │
    │   • Create simulation components    │
    │   • Setup restraints & templates    │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │     PHASE 2: ENDSTATE SIMS          │
    │   run_endstate_simulations()        │
    │   • Complex MD simulation           │
    │   • Receptor MD simulation          │
    │   • Ligand MD simulation            │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │   PHASE 3: DECOMPOSITION            │
    │   decompose_system_and_generate_    │
    │   restraints()                      │
    │   • Split complex → receptor+ligand │
    │   • Generate Boresch restraints     │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  PHASE 4: INTERMEDIATE SETUP        │
    │   setup_intermediate_simulations()  │
    │   • Setup complex intermediates     │
    │   • Setup receptor intermediates    │
    │   • Setup ligand intermediates      │
    │   • Setup flat-bottom simulations   │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  PHASE 5: RUN INTERMEDIATES         │
    │   run_intermediate_simulations()    │
    │   • Execute complex MD simulations  │
    │   • Execute receptor MD simulations │
    │   • Execute ligand MD simulations   │
    │   • Execute flat-bottom simulations │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │   PHASE 6: ENERGY ANALYSIS          │
    │   run_post_analysis_intermediate_   │
    │   simulations()                     │
    │   • Post-process complex energies   │
    │   • Post-process receptor energies  │
    │   • Post-process ligand energies    │
    │   • Post-process flat-bottom energies│
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  PHASE 7: FREE ENERGY & CONSOLIDATE │
    │   compute_free_energy_and_          │
    │   consolidate()                     │
    │   • Run MBAR analysis               │
    │   • Run exponential averaging       │
    │   • Consolidate results             │
    └─────────────────────────────────────┘
                           │
                           ▼
                    🎉 WORKFLOW COMPLETE
```

## Detailed Module Interaction Flow

```
implicit_ddm_workflow.py
        │
        ├── workflow_phases.py
        │   ├── setup_workflow_components()
        │   ├── run_endstate_simulations()
        │   ├── decompose_system_and_generate_restraints()
        │   ├── setup_intermediate_simulations()
        │   ├── run_intermediate_simulations()
        │   ├── run_post_analysis_intermediate_simulations()
        │   ├── compute_free_energy_and_consolidate()
        │   └── initilized_jobs() [Progress tracking]
        │
        ├── simulations.py [Called within phases]
        │   ├── Simulation class
        │   ├── Calculation class
        │   └── MD execution logic
        │
        ├── runner.py [Called within phases]
        │   ├── IntermidateRunner class
        │   ├── Job orchestration
        │   └── Post-processing logic
        │
        ├── setup_simulations.py [Called within phases]
        │   ├── SimulationSetup class
        │   └── Parameter setup logic
        │
        ├── restraints.py [Called within phases]
        │   ├── RestraintMaker class
        │   └── Boresch restraint generation
        │
        └── config.py [Used throughout]
            ├── Config class
            ├── SystemSettings class
            └── Configuration management
```

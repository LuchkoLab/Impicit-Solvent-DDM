.. _ddm_workflow-label:

DDM Workflow Structure
======================

This document shows the structural flow of the Double Decoupling Method (DDM) workflow as implemented in ``implicit_ddm_workflow.py``.

Overview
--------

The DDM workflow is organized into seven distinct phases, each handling a specific aspect of the binding free energy calculation. The workflow automates the entire process from initial setup through final analysis and result consolidation.

Simplified Workflow Flow
------------------------

The following diagram shows the high-level flow of the DDM workflow:

.. code-block:: text

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

Workflow Phases
---------------

**Phase 1: Setup**
   Initial setup of workflow components, including creation of simulation components and setup of restraints and templates.

**Phase 2: Endstate Simulations**
   Execution of the three fundamental simulations:
   
   - Complex MD simulation (bound state)
   - Receptor MD simulation (unbound receptor)
   - Ligand MD simulation (unbound ligand)

**Phase 3: Decomposition**
   System decomposition and restraint generation:
   
   - Split the complex into receptor and ligand components
   - Generate Boresch orientational restraints

**Phase 4: Intermediate Setup**
   Preparation of intermediate simulation states:
   
   - Setup complex intermediate states
   - Setup receptor intermediate states
   - Setup ligand intermediate states
   - Setup flat-bottom restraint simulations

**Phase 5: Run Intermediates**
   Execution of all intermediate simulations to sample the thermodynamic pathway.

**Phase 6: Energy Analysis**
   Post-processing of simulation data:
   
   - Extract and analyze energies from all simulations
   - Prepare data for free energy calculations

**Phase 7: Free Energy & Consolidate**
   Final analysis and result consolidation:
   
   - MBAR analysis for free energy calculations
   - Exponential averaging calculations
   - Consolidation of all results

Detailed Module Interaction Flow
--------------------------------

The following diagram shows the detailed module structure and interactions:

.. code-block:: text

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

Module Descriptions
-------------------

**implicit_ddm_workflow.py**
   Main workflow orchestrator that coordinates all phases of the DDM calculation.

**workflow_phases.py**
   Contains the implementation of all seven workflow phases and progress tracking.

**simulations.py**
   Core simulation classes and MD execution logic.

**runner.py**
   Job orchestration and post-processing functionality.

**setup_simulations.py**
   Simulation setup and parameter configuration.

**restraints.py**
   Boresch restraint generation and management.

**config.py**
   Configuration management and system settings.

Related Documentation
---------------------

- :ref:`Implementation Details <ddm_cycle-label>` - Detailed explanation of the thermodynamic cycle
- :ref:`API Documentation <api-label>` - Complete API reference for all modules

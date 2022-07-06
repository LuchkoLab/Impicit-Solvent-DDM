



import copy
import os
import os.path
import re
import shutil
import string
import sys
from argparse import ArgumentParser
from mimetypes import init
from pathlib import Path
from posixpath import split
from tokenize import Name
from typing import List, Optional, Type

import numpy as np
import yaml
from click import option
from pyparsing import opAssoc
from setuptools import setup
from toil.common import Toil
from toil.job import Job, JobFunctionWrappingJob

# from implicit_solvent_ddm.alchemical import (get_intermidate_parameter_files,
#                                              split_complex)
from alchemical import get_intermidate_parameter_files, split_complex_system
# from implicit_solvent_ddm.config import Config
from config import Config
#from implicit_solvent_ddm.mdin import get_mdins
# from implicit_solvent_ddm.remd import remd_workflow, run_minimization
from mdin import get_mdins
# from implicit_solvent_ddm.restraints import (get_conformational_restraints,
#                                              get_flat_bottom_restraints,
#                                              get_orientational_restraints,
#                                              write_empty_restraint)
from restraints import (get_conformational_restraints,
                        get_flat_bottom_restraints,
                        get_orientational_restraints, write_empty_restraint,
                        write_restraint_forces)
#from implicit_solvent_ddm.simulations import REMDSimulation, Simulation
from simulations import ExtractTrajectories, REMDSimulation, Simulation

#local imports 



#from implicit_solvent_ddm.remd import run_remd
 
 
def ddm_workflow(job:JobFunctionWrappingJob, config:Config, inptraj_ID = None, solute = "system", dirstuct_traj_args={}, post_process=False):
    '''
    Double decoupling workflow 

    Runs long simulations at the end states (receptor, ligand & complex). 
    Creates orientational and conformational restraints for intermediate states (short simulation runs). 

    Parameters
    ----------
    toil: class toil.common.Toil
        A contect manager that represents a Toil workflow 
    df_config_inputs: pandas.DataFrame 
        A data frame containing user's config parameters and imported Toil fileID's 
    argSet: dict
        Dictionary containing user's config parameters 
    work_dir: str 
        User's initial working path directory 

    Returns
    -------
    end_state_job: toil.job.JobFunctionWrappingJob
        contains the entire workflow in indiviudual jobs. 
    '''
    # temp_dir = job.fileStore.getLocalTempDir()
    
   
    calc_list = []
    data_frames = []
    
    #post process setup workflow 
    if post_process:
        inptraj_ID = inptraj_ID
        workflow = config.workflow.update_worflow(solute)
        ligand_receptor_dirstruct = "post_process_apo"
        complex_dirstruct = "post_process_halo"
        mdin_no_solv = config.inputs["post_nosolv_mdin"]
        default_mdin = config.inputs["post_mdin"]
        md_jobs = job.addChildJobFn(initilized_jobs)
 
    else:
        workflow = config.workflow
        ligand_receptor_dirstruct = "dirstruct_apo"
        complex_dirstruct = "dirstruct_halo"
        
        setup_inputs = job.wrapJobFn(get_intermidate_parameter_files, 
                                                        config.endstate_files.complex_parameter_filename, config.endstate_files.complex_coordinate_filename,
                                                        config.amber_masks.ligand_mask, config.amber_masks.receptor_mask)
        # Add inputs as first child to root job
        job.addChild(setup_inputs)
        
        config.inputs["ligand_no_charge_parm_ID"] = setup_inputs.rv(0)
        config.inputs["complex_ligand_no_charge_ID"] = setup_inputs.rv(1)
        config.inputs["complex_no_ligand_interaction_ID"] = setup_inputs.rv(2)
        
        #fill in intermidate mdin     
        mdins = setup_inputs.addChildJobFn(get_mdins, config.intermidate_args.mdin_intermidate_config)
        
        #set intermidate mdin files 
        config.inputs["default_mdin"] = mdins.rv(0)
        config.inputs["no_solvent_mdin"] = mdins.rv(1)
        config.inputs["post_mdin"] = mdins.rv(2)
        config.inputs["post_nosolv_mdin"] = mdins.rv(3)
        default_mdin = mdins.rv(0)
        mdin_no_solv = mdins.rv(1)
        #write empty restraint.RST 
        empty_restraint = setup_inputs.addChildJobFn(write_empty_restraint)
        config.inputs["empty_restraint"] = empty_restraint.rv()
        #flat bottom restraints potential restraints 
        flat_bottom_template = setup_inputs.addChildJobFn(get_flat_bottom_restraints, 
                                                    config.endstate_files.complex_parameter_filename, config.endstate_files.complex_coordinate_filename,
                                                    config.endstate_method.flat_bottom_restraints)
        
        config.inputs["flat_bottom_restraint"] = flat_bottom_template.rv()
        
        
        #Begin running END State Simulations
        #config.workflow.run_endstate_method and post_process==False: 
        
        if workflow.run_endstate_method:
            endstate_method = setup_inputs.addFollowOnJobFn(initilized_jobs)
    
            if config.endstate_method.endstate_method_type == 'remd':
               
                #run endstate method for complex system
                minimization_complex = endstate_method.addChild(Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                            config.num_cores_per_system.complex_ncores, 
                                                            config.endstate_files.complex_parameter_filename, config.endstate_files.complex_coordinate_filename,
                                                            config.inputs["min_mdin"], config.inputs["flat_bottom_restraint"], 
                                                            {"runtype": 'minimization', "filename": "min", "topology": config.endstate_files.complex_parameter_filename}))
                
                equilibrate_complex = minimization_complex.addFollowOn(REMDSimulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                                                config.endstate_method.remd_args.nthreads, 
                                                                                config.endstate_files.complex_parameter_filename, minimization_complex.rv(0),
                                                                                config.endstate_method.remd_args.equilibration_replica_mdins,
                                                                                config.inputs["flat_bottom_restraint"], "equil", 
                                                                                config.endstate_method.remd_args.ngroups, 
                                                                                {"runtype": 'equilibration', "topology": config.endstate_files.complex_parameter_filename}))
                
                remd_complex = equilibrate_complex.addFollowOn(REMDSimulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                                            config.endstate_method.remd_args.nthreads, config.endstate_files.complex_parameter_filename,
                                                                            equilibrate_complex.rv(0), config.endstate_method.remd_args.remd_mdins,
                                                                            config.inputs["flat_bottom_restraint"], 'remd', config.endstate_method.remd_args.ngroups,
                                                                            {"runtype": 'remd', "topology": config.endstate_files.complex_parameter_filename}))
                
                #extact target temparture trajetory and last frame 
                extract_complex = remd_complex.addFollowOn(ExtractTrajectories(config.endstate_files.complex_parameter_filename, remd_complex.rv(1),
                                                                                config.endstate_method.remd_args.target_temperature))
                
                config.inputs["endstate_complex_traj"] = extract_complex.rv(0)
                config.inputs["endstate_complex_lastframe"] = extract_complex.rv(1)
                #run minimization at the end states for ligand system only 
                minimization_ligand = endstate_method.addChild(Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                            config.num_cores_per_system.ligand_ncores, 
                                                            config.endstate_files.ligand_parameter_filename, config.endstate_files.ligand_coordinate_filename,
                                                            config.inputs["min_mdin"], config.inputs["empty_restraint"], 
                                                            {"runtype": 'minimization', "filename": "min", "topology": config.endstate_files.ligand_parameter_filename}))
                
                equilibrate_ligand = minimization_ligand.addFollowOn(REMDSimulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                                                config.endstate_method.remd_args.nthreads, 
                                                                                config.endstate_files.ligand_parameter_filename, minimization_ligand.rv(0),
                                                                                config.endstate_method.remd_args.equilibration_replica_mdins,
                                                                                config.inputs["empty_restraint"], "equil", 
                                                                                config.endstate_method.remd_args.ngroups, 
                                                                                {"runtype": 'equilibration', "topology": config.endstate_files.ligand_parameter_filename}))
                
                remd_ligand = equilibrate_ligand.addFollowOn(REMDSimulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                                            config.endstate_method.remd_args.nthreads, config.endstate_files.ligand_parameter_filename,
                                                                            equilibrate_ligand.rv(0), config.endstate_method.remd_args.remd_mdins,
                                                                            config.inputs["empty_restraint"], 'remd', config.endstate_method.remd_args.ngroups,
                                                                            {"runtype": 'remd', "topology": config.endstate_files.ligand_parameter_filename}))
                #extact target temparture trajetory and last frame 
                extract_ligand = remd_ligand.addFollowOn(ExtractTrajectories(config.endstate_files.ligand_parameter_filename, remd_ligand.rv(1),
                                                                                config.endstate_method.remd_args.target_temperature))
                config.inputs["endstate_ligand_traj"] = extract_ligand.rv(0)
                config.inputs["endstate_ligand_lastframe"] = extract_ligand.rv(1)
               
                # minimization_receptor = endstate_method.addChild(Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                #                                             config.num_cores_per_system.receptor_ncores, 
                #                                             config.endstate_files.receptor_parameter_filename, config.endstate_files.receptor_coordinate_filename,
                #                                             config.inputs["min_mdin"], config.inputs["empty_restraint"], 
                #                                             {"runtype": 'minimization', "filename": "min", "topology": config.endstate_files.receptor_parameter_filename}))
        # user ran there own endstate calculation just split the coordinate 
        else:
            endstate_method = setup_inputs.addFollowOn(ExtractTrajectories(config.endstate_files.complex_parameter_filename, 
                                                                           config.endstate_files.complex_coordinate_filename))
            config.inputs["endstate_complex_traj"] = endstate_method.rv(0)
            config.inputs["endstate_complex_lastframe"] = endstate_method.rv(1)
            ligand_extract = endstate_method.addChild(ExtractTrajectories(config.endstate_files.ligand_parameter_filename, 
                                                                           config.endstate_files.ligand_coordinate_filename))
            config.inputs["endstate_ligand_traj"] = ligand_extract.rv(0)
            
            
               
        
        
        #split the complex into host and substrate using the endstate lastframe  
        split_job = endstate_method.addFollowOnJobFn(split_complex_system, 
                                                            config.endstate_files.complex_parameter_filename, config.inputs["endstate_complex_lastframe"], 
                                                            config.amber_masks.ligand_mask, config.amber_masks.receptor_mask)
            
        config.inputs["ligand_endstate_frame"] = split_job.rv(1)
        config.inputs["receptor_endstate_frame"] = split_job.rv(0)
        
        # create orientational and conformational restraints using the lastframe of the complex endstate simulation 
        conformational_restraints = split_job.addChildJobFn(get_conformational_restraints, 
                                                    config.endstate_files.complex_parameter_filename, config.inputs["endstate_complex_lastframe"],
                                                    config.amber_masks.receptor_mask, config.amber_masks.ligand_mask)
        orientational_restraints = conformational_restraints.addFollowOnJobFn(get_orientational_restraints, 
                                                    config.endstate_files.complex_parameter_filename, config.inputs["endstate_complex_lastframe"],
                                                    config.amber_masks.receptor_mask, config.amber_masks.ligand_mask,
                                                    config.intermidate_args.restraint_type)
        
        #fill in orientational and conformational forces within templates 
        for (conformational_force, orientational_force) in zip(config.intermidate_args.conformational_restraints_forces, config.intermidate_args.orientational_restriant_forces):
            
            config.inputs[f"ligand_{conformational_force}_rst"] = orientational_restraints.addChildJobFn(write_restraint_forces, conformational_restraints.rv(1), conformational_force=conformational_force).rv()
            
            
            config.inputs[f"ligand_{conformational_force}_rst"] = orientational_restraints.addChildJobFn(write_restraint_forces,
                                                                                                            conformational_restraints.rv(1), conformational_force=conformational_force).rv()
            config.inputs[f"receptor_{conformational_force}_rst"] = orientational_restraints.addChildJobFn(write_restraint_forces,
                                                                                                            conformational_restraints.rv(2), conformational_force=conformational_force).rv()
            config.inputs[f"complex_{conformational_force}_{orientational_force}_rst"] = orientational_restraints.addChildJobFn(write_restraint_forces, 
                                                                                                                    conformational_restraints.rv(0),
                                                                                                                    orientational_restraints.rv(),
                                                                                                                    conformational_force=conformational_force,
                                                                                                                    orientational_force=orientational_force).rv()
        md_jobs = orientational_restraints.addFollowOnJobFn(initilized_jobs)
        # split the complex coordinates into ligand and receptor systems once endstate simulation is completed. 
        # post process endstates once the restraint files are created     
        complex_endstate_post_workflow = md_jobs.addChildJobFn(ddm_workflow, 
                                    config, 
                                    inptraj_ID = [config.inputs["endstate_complex_traj"]], 
                                    solute = 'complex',
                                    dirstuct_traj_args= {
                                    "traj_state_label": 8,
                                    "traj_igb": config.intermidate_args.igb_solvent,
                                    "trajectory_restraint_conrest": 0.0,
                                    "trajectory_restraint_orenrest": 0.0,
                                    "filename": "state_8_endstate_postprocess",
                                    "runtype": f"Running post process with trajectory: {config.inputs['endstate_complex_traj']}"}.copy(), post_process = True)
        
        ligand_endstate_post_workflow = md_jobs.addChildJobFn(ddm_workflow, 
                                    config, 
                                    inptraj_ID = [config.inputs["endstate_ligand_traj"]], 
                                    solute = 'ligand',
                                    dirstuct_traj_args= {"traj_state_label": 2,
                                    "traj_igb": config.intermidate_args.igb_solvent,
                                    "filename": "state_2_endstate_postprocess",
                                    "trajectory_restraint": 0.0,
                                    "runtype": f"Running post process with trajectory: {config.inputs['endstate_ligand_traj']}"}.copy(), post_process = True)
        
        #begin running postprocess for endstate trajectories 
        complex_endstate_post_completed = complex_endstate_post_workflow.addFollowOnJobFn(run_post_process, complex_endstate_post_workflow.rv())
        ligand_endstate_post_completed = ligand_endstate_post_workflow.addFollowOnJobFn(run_post_process, ligand_endstate_post_workflow.rv())
        
        #parse data 

    if workflow.end_state_postprocess: 
        state_label = 2
        restraint_file = config.inputs["empty_restraint"]
        dirstruct = "post_process_apo"
        if solute == 'ligand':
            topology = config.endstate_files.ligand_parameter_filename
            coordinate = config.endstate_files.ligand_coordinate_filename
            num_cores = config.num_cores_per_system.ligand_ncores
        elif solute == 'receptor':
            topology = config.endstate_files.receptor_parameter_filename
            coordinate = config.endstate_files.receptor_coordinate_filename
            num_cores = config.num_cores_per_system.receptor_ncores
        else:
            topology = config.endstate_files.complex_parameter_filename
            coordinate = config.endstate_files.complex_coordinate_filename 
            num_cores = config.num_cores_per_system.complex_ncores
            dirstruct = "post_process_halo"
            state_label = 8
            restraint_file = config.inputs["flat_bottom_restraint"]
        end_state_args = {
            "topology": topology,
            "state_label": state_label,
            "igb": config.intermidate_args.igb_solvent,
            "conformational_restraint": 0.0,
            "orientational_restraints": 0.0, 
        }
        end_state_args.update(dirstuct_traj_args)
        end_state_prod = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        num_cores, topology, coordinate,
                                                        config.inputs["post_mdin"],  
                                                        restraint_file, 
                                                        directory_args=end_state_args.copy(),
                                                        dirstruct = dirstruct, inptraj= inptraj_ID)
        calc_list.append(end_state_prod)
    
    #define max conformational and restraint forces
    max_con_force = max(config.intermidate_args.conformational_restraints_forces)
    max_orien_force = max(config.intermidate_args.orientational_restriant_forces)
    
    max_con_exponent = max(config.intermidate_args.exponent_conformational_forces)
    max_orien_exponent = max(config.intermidate_args.exponent_orientational_forces)
    
    #turning off the solvent for ligand simulation, set max force of conformational restraints    
    if workflow.remove_GB_solvent_ligand:
        no_solv_args = {
            "topology": config.endstate_files.ligand_parameter_filename, 
            "state_label": '4', 
            "conformational_restraint": max_con_exponent, 
            "igb": "igb_6",
            "filename": "state_4_prod",
            "runtype": "Running production Simulation in state 4"
        }
        no_solv_args.update(dirstuct_traj_args)
        no_solv_ligand = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.ligand_ncores, config.endstate_files.ligand_parameter_filename,
                                                        config.inputs["ligand_endstate_frame"], mdin_no_solv,  
                                                        config.inputs[f"ligand_{max_con_force}_rst"], 
                                                        directory_args=no_solv_args.copy(),
                                                        dirstruct = ligand_receptor_dirstruct, inptraj= inptraj_ID)
        #md_jobs.addChild(no_solv_ligand)
        if not post_process:
            md_jobs.addChild(no_solv_ligand)
            post_process_ligand = no_solv_ligand.addFollowOnJobFn(ddm_workflow, 
                                                                  config, 
                                                                  inptraj_ID = no_solv_ligand.rv(1), 
                                                                  solute = 'ligand',
                                                                  dirstuct_traj_args= {"traj_state_label": 4,
                                                                                       "trajectory_restraint": max_con_exponent,
                                                                                       "traj_igb": "igb_6",
                                                                                       "filename": "state_4_postprocess",
                                                                                       "runtype": f"Running post process with trajectory: {no_solv_ligand.rv(1)}"}.copy(), post_process = True)
            post_process_ligand.addFollowOnJobFn(run_post_process, post_process_ligand.rv())
        
        else: 
            calc_list.append(no_solv_ligand)
            
        
    #set ligand overall charge to 0 
    #if config.workflow.remove_ligand_charges:
    if workflow.remove_ligand_charges:
        ligand_no_charge_args = {
            "topology": config.endstate_files.ligand_parameter_filename, 
            "state_label": '5', 
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_5_prod",
            "runtype": "Production Simulation in State 5"
        }
        ligand_no_charge_args.update(dirstuct_traj_args)
        
        ligand_no_charge = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                         config.num_cores_per_system.ligand_ncores, config.inputs["ligand_no_charge_parm_ID"], 
                                                         config.inputs["ligand_endstate_frame"], mdin_no_solv, 
                                                         config.inputs[f"ligand_{max_con_force}_rst"], 
                                                         directory_args=ligand_no_charge_args, 
                                                         dirstruct = ligand_receptor_dirstruct, inptraj= inptraj_ID)
        # ligand_jobs.addChild(ligand_no_charge)
        
        if not post_process:
            md_jobs.addChild(ligand_no_charge)
            post_no_charge = ligand_no_charge.addFollowOnJobFn(ddm_workflow, config, 
                                              inptraj_ID = ligand_no_charge.rv(1), 
                                              solute = 'ligand',
                                              dirstuct_traj_args= {"traj_state_label": '5',
                                                                   "trajectory_restraint": max_con_exponent,
                                                                   "traj_igb": "igb_6",
                                                                   "filename": "state_5_postprocess",
                                                                   "runtype": f"Running post process with trajectory: {ligand_no_charge.rv(1)}"}.copy(), post_process = True)
            
            post_no_charge.addFollowOnJobFn(run_post_process, post_no_charge.rv())
        
        else:
            calc_list.append(ligand_no_charge)
            
            
    #Desolvation of receptor 
    #if config.workflow.remove_GB_solvent_receptor: 
    if not workflow.ignore_receptor:
        no_solv_args_receptor = {
            "topology": config.endstate_files.receptor_parameter_filename, 
            "state_label": 4, 
            "conformational_restraint": max_con_exponent,
            "igb": "igb_6",
            "filename": "state_4_prod",
            "runtype": "Running production simulation in state 4: Receptor only"
        }
        
        no_solv_args_receptor.update(dirstuct_traj_args)
        
        no_solv_receptor = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.receptor_ncores, config.endstate_files.receptor_parameter_filename,
                                                        config.inputs["receptor_endstate_frame"], mdin_no_solv,  
                                                        config.inputs[f"receptor_{max_con_force}_rst"], 
                                                        directory_args= no_solv_args_receptor,
                                                        dirstruct = ligand_receptor_dirstruct, inptraj= inptraj_ID)
        #receptor_jobs.addChild(no_solv_receptor)
        if not post_process:
            md_jobs.addChild(no_solv_receptor)
            post_no_solv_receptor = no_solv_receptor.addFollowOnJobFn(ddm_workflow, config, 
                                              inptraj_ID = no_solv_receptor.rv(1),
                                              solute = 'receptor',
                                              dirstuct_traj_args = {"traj_state_label": 4,
                                                                    "trajectory_restraint": max_con_exponent,
                                                                    "traj_igb": "igb_6",
                                                                    "filename": "state_4_postprocess",
                                                                    "runtype": f"Running post process with trajectory: {no_solv_receptor.rv(1)}"}.copy(), post_process = True)
            
            post_no_solv_receptor.addFollowOnJobFn(run_post_process, post_no_solv_receptor.rv())   
        else:
            calc_list.append(no_solv_receptor)
            
    
    #Complex simulations 
    # Exclusions turned on, no electrostatics, in gas phase 
    # if config.workflow.complex_ligand_exclusions:
    if workflow.complex_ligand_exclusions:
        complex_ligand_exclusions_args = {"topology": config.endstate_files.complex_parameter_filename, 
                                          "state_label": '7', 
                                          "igb":  "igb_6",
                                          "conformational_restraint": max_con_exponent,
                                          "orientational_restraints": max_orien_exponent,
                                          "filename": "state_7_prod",
                                          "runtype": "Running production simulation in state 7: Complex"}
        
        complex_ligand_exclusions_args.update(dirstuct_traj_args)
        
        complex_no_interactions = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.complex_ncores, config.inputs["complex_no_ligand_interaction_ID"],
                                                        config.inputs["endstate_complex_lastframe"], mdin_no_solv,  
                                                        config.inputs[f"complex_{max_con_force}_{max_orien_force}_rst"], 
                                                        directory_args=complex_ligand_exclusions_args,
                                                        dirstruct=complex_dirstruct, inptraj= inptraj_ID)
        if not post_process:
            md_jobs.addChild(complex_no_interactions)
            
            complex_no_interactions_post = complex_no_interactions.addFollowOnJobFn(ddm_workflow, config, 
                                              inptraj_ID = complex_no_interactions.rv(1),
                                              solute = 'complex',
                                              dirstuct_traj_args = {"traj_state_label": 7,
                                                                    "trajectory_restraint_conrest": max_con_exponent,
                                                                    "trajectory_restraint_orenrest": max_orien_exponent,
                                                                    "traj_igb": "igb_6",
                                                                    "filename": "state_7_postprocess",
                                                                    "runtype": f"Running post process with trajectory: {complex_no_interactions.rv(1)}"}.copy(), 
                                              post_process = True)
            
            complex_no_interactions_post.addFollowOnJobFn(run_post_process, complex_no_interactions_post.rv())
        else:
            job.log(f"update complex args {complex_ligand_exclusions_args}")
            calc_list.append(complex_no_interactions)
            
    # No electrostatics and in the gas phase
    if workflow.complex_turn_off_exclusions:
        complex_turn_off_exclusions_args = {
            "topology": config.endstate_files.complex_parameter_filename, 
            "state_label": '7a', 
            "igb":  "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7a_prod",
            "runtype": "Running production simulation in state 7a: Complex"
        }
        complex_turn_off_exclusions_args.update(dirstuct_traj_args)
        
        complex_no_electrostatics = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.complex_ncores, config.inputs["complex_ligand_no_charge_ID"],
                                                        config.inputs["endstate_complex_lastframe"], mdin_no_solv,  
                                                        config.inputs[f"complex_{max_con_force}_{max_orien_force}_rst"], 
                                                        directory_args = complex_turn_off_exclusions_args,
                                                        dirstruct = complex_dirstruct, inptraj= inptraj_ID)
        if not post_process:
            md_jobs.addChild(complex_no_electrostatics)
        
            complex_no_electrostatics_post = complex_no_electrostatics.addFollowOnJobFn(ddm_workflow, config, 
                                                inptraj_ID = complex_no_electrostatics.rv(1),
                                                solute = 'complex',
                                                dirstuct_traj_args = {"traj_state_label": '7a',
                                                                    "trajectory_restraint_conrest": max_con_exponent,
                                                                    "trajectory_restraint_orenrest": max_orien_exponent,
                                                                    "traj_igb": "igb_6",
                                                                    "filename": "state_7a_postprocess",
                                                                    "runtype": f"Running post process with trajectory: {complex_no_electrostatics.rv(1)}"}.copy(), 
                                                post_process = True)
            
            complex_no_electrostatics_post.addFollowOnJobFn(run_post_process, complex_no_electrostatics_post.rv())
        else:
            job.log(f"update complex_turn_off_exclusions_args {complex_turn_off_exclusions_args}")
            calc_list.append(complex_no_electrostatics)
        
    # Turn ligand charges and in the gas phase  

    if workflow.complex_turn_on_ligand_charges:
        complex_turn_on_ligand_charges_args = {
            "topology": config.endstate_files.complex_parameter_filename, 
            "state_label": '7b', 
            "igb":  "igb_6",
            "conformational_restraint": max_con_exponent,
            "orientational_restraints": max_orien_exponent,
            "filename": "state_7b_prod",
            "runtype": "Running production simulation in state 7b: Complex"
        }
        complex_turn_on_ligand_charges_args.update(dirstuct_traj_args)
        
        complex_turn_on_ligand_charges = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.complex_ncores, config.endstate_files.complex_parameter_filename,
                                                        config.inputs["endstate_complex_lastframe"], mdin_no_solv,  
                                                        config.inputs[f"complex_{max_con_force}_{max_orien_force}_rst"], 
                                                        directory_args=complex_turn_on_ligand_charges_args,
                                                        dirstruct=complex_dirstruct, inptraj= inptraj_ID)
        if not post_process:
            md_jobs.addChild(complex_turn_on_ligand_charges)
            complex_turn_on_ligand_charges_post = complex_turn_on_ligand_charges.addFollowOnJobFn(ddm_workflow, 
                                                                                                  config, 
                                                                                                  inptraj_ID = complex_turn_on_ligand_charges.rv(1),
                                                                                                  solute = 'complex',
                                                                                                  dirstuct_traj_args = {"traj_state_label": '7b',
                                                                                                                        "trajectory_restraint_conrest": max_con_exponent,
                                                                                                                        "trajectory_restraint_orenrest": max_orien_exponent,
                                                                                                                        "traj_igb": "igb_6",
                                                                                                                        "filename": "state_7b_postprocess",
                                                                                                                        "runtype": f"Running post process with trajectory: {complex_turn_on_ligand_charges.rv(1)}"}.copy(), 
                                                                                                  post_process = True)
            
            complex_turn_on_ligand_charges_post.addFollowOnJobFn(run_post_process, complex_turn_on_ligand_charges_post.rv())
        
        else:
            calc_list.append(complex_turn_on_ligand_charges)
    
    #Lambda window interate through conformational and orientational restraint forces     
    for (con_force, orien_force) in zip(config.intermidate_args.conformational_restraints_forces, config.intermidate_args.orientational_restriant_forces):
        #add conformational restraints 
        #if config.workflow.add_ligand_conformational_restraints:
        exponent_conformational = np.log2(con_force) 
        exponent_orientational = np.log2(orien_force)
        if workflow.add_ligand_conformational_restraints:
            ligand_window_args = {
                "topology": config.endstate_files.ligand_parameter_filename, 
                "state_label": 2,
                "conformational_restraint": exponent_conformational, 
                "igb": f"igb_{config.intermidate_args.igb_solvent}",
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, Conformational restraint: {con_force}"
            }
            ligand_window_args.update(dirstuct_traj_args)
            
            ligand_windows = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.ligand_ncores, config.endstate_files.ligand_parameter_filename,
                                                        config.inputs["ligand_endstate_frame"], default_mdin,  
                                                        config.inputs[f"ligand_{con_force}_rst"], 
                                                        directory_args=ligand_window_args,
                                                        dirstruct= ligand_receptor_dirstruct,
                                                        inptraj= inptraj_ID)
            if not post_process:
                md_jobs.addChild(ligand_windows)
                ligand_windows_post = ligand_windows.addFollowOnJobFn(ddm_workflow, 
                                                                      config, 
                                                                      inptraj_ID = ligand_windows.rv(1),
                                                                      solute = 'ligand',
                                                                      dirstuct_traj_args = {
                                                                        "traj_state_label": 2,
                                                                        "trajectory_restraint": exponent_conformational,
                                                                        "traj_igb": f"igb_{config.intermidate_args.igb_solvent}",
                                                                        "filename": "state_2_postprocess",
                                                                        "runtype": f"Running post process with trajectory: {ligand_windows.rv(1)}"}.copy(), post_process = True)
                
                ligand_windows_post.addFollowOnJobFn(run_post_process, ligand_windows_post.rv())
            else:
                calc_list.append(ligand_windows)
                
        
        if not workflow.ignore_receptor:
            
            receptor_window_args = {
                "topology": config.endstate_files.receptor_parameter_filename, 
                "state_label": 2,
                "conformational_restraint": exponent_conformational, 
                "igb": config.intermidate_args.igb_solvent,
                "filename": f"state_2_{con_force}_prod",
                "runtype": f"Running restraint window, conformational restraint: {con_force}"}
            
            receptor_window_args.update(dirstuct_traj_args)
            
            receptor_windows = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.receptor_ncores, config.endstate_files.receptor_parameter_filename,
                                                        config.inputs["receptor_endstate_frame"], default_mdin,  
                                                        config.inputs[f"receptor_{con_force}_rst"], 
                                                        directory_args=receptor_window_args,
                                                        dirstruct=ligand_receptor_dirstruct,
                                                        inptraj= inptraj_ID)
            if not post_process:
                md_jobs.addChild(receptor_windows)
                receptor_windows_post = receptor_windows.addFollowOnJobFn(ddm_workflow, 
                                                                      config, 
                                                                      inptraj_ID = receptor_windows.rv(1),
                                                                      solute = 'receptor',
                                                                      dirstuct_traj_args = {
                                                                        "traj_state_label": 2,
                                                                        "trajectory_restraint": exponent_conformational,
                                                                        "traj_igb": config.intermidate_args.igb_solvent,
                                                                        "filename": "state_2_postprocess",
                                                                        "runtype": f"Running post process with trajectory: {receptor_windows.rv(1)}"}.copy(), post_process = True)
                
                receptor_windows_post.addFollowOnJobFn(run_post_process, receptor_windows_post.rv())
            
            else:
                calc_list.append(receptor_windows)
              
        
        # slowly remove conformational and orientational restraints  
        # if config.workflow.complex_remove_restraint:
        if workflow.complex_remove_restraint:
            remove_restraints_args = {
                "topology": config.endstate_files.complex_parameter_filename, 
                "state_label": 8, 
                "igb": config.intermidate_args.igb_solvent,
                "conformational_restraint": exponent_conformational,
                "orientational_restraints": orien_force,
                "filename": f"state_8_{con_force}_{orien_force}_prod",
                "runtype": f"Running restraint window. Conformational restraint: {con_force} and orientational restraint: {orien_force}"}
            
            remove_restraints_args.update(dirstuct_traj_args)
            
            remove_restraints = Simulation(config.system_settings.executable, config.system_settings.mpi_command,
                                                        config.num_cores_per_system.complex_ncores, config.endstate_files.complex_parameter_filename,
                                                        config.inputs["endstate_complex_lastframe"], default_mdin,  
                                                        config.inputs[f"complex_{con_force}_{orien_force}_rst"], 
                                                         directory_args=remove_restraints_args,
                                                        dirstruct= complex_dirstruct, inptraj= inptraj_ID)
            if not post_process:
                md_jobs.addChild(remove_restraints)
                remove_restraints_windows_post = remove_restraints.addFollowOnJobFn(ddm_workflow, 
                                                                      config, 
                                                                      inptraj_ID = remove_restraints.rv(1),
                                                                      solute = 'complex',
                                                                      dirstuct_traj_args = {
                                                                        "traj_state_label": 8,
                                                                        "trajectory_restraint_conrest": exponent_conformational,
                                                                        "trajectory_restraint_orenrest": exponent_orientational,
                                                                        "traj_igb": config.intermidate_args.igb_solvent,
                                                                        "filename": "state_8_postprocess",
                                                                        "runtype": f"Running post process with trajectory: {remove_restraints.rv(1)}"}.copy(), 
                                                                      post_process = True)
                
                remove_restraints_windows_post.addFollowOnJobFn(run_post_process, remove_restraints_windows_post.rv())
            
            else:
                calc_list.append(remove_restraints)
        
    if post_process:
        return calc_list
    #PARSE DATA AND DO 
def return_trajectories(job, *args):
    for arg in args:
        job.log(f'arg: {arg}')
    return args
       
def run_post_process(job, sims):
    
    output_data = []
    for sim in sims:
        output_sims = job.addChild(sim)
        
    #     # parse_data = output_sims.addFollowOnJobFn(parse_data, 
    #     output_data.append((output_sims.rv(2), output_sims.rv(3)))
    
    # return output_data    
    #parse in data and create pandas dataframe   


    
        
def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return 

def main():
    
    # parser = Job.Runner.getDefaultArgumentParser()
    # parser.add_argument('--config_file', nargs='*', type=str, required=True, help="configuartion file with input parameters")
    # parser.add_argument("--ignore_receptor", action= "store_true", help=" Receptor MD caluculations with not be performed.")
    # options = parser.parse_args()
    # options.logLevel = "INFO"
    # options.clean = "always"
    # config_file = options.config_file[0]
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    yaml_file = '/home/ayoub/nas0/Impicit-Solvent-DDM/new_workflow.yaml'
    try:
        with open(yaml_file) as f:
            config_file = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
        
    config = Config.from_config(config_file)
    
    if options.workDir:
        config.system_settings.working_directory = os.path.abspath(options.workDir)
    else:
        config.system_settings.working_directory = os.getcwd()

    config.ignore_receptor = False 
    
    if config.workflow.run_endstate_method: 
        if not os.path.exists(os.path.join(config.system_settings.working_directory, "mdgb/structs/ligand")):
            os.makedirs(os.path.join(config.system_settings.working_directory, "mdgb/structs/ligand"))
        if not os.path.exists(os.path.join(config.system_settings.working_directory, "mdgb/structs/receptor")):
            os.makedirs(os.path.join(config.system_settings.working_directory, "mdgb/structs/receptor"))
    
    config.get_receptor_ligand_topologies()
    
    #create a log file
    job_number = 1
    while os.path.exists(f"mdgb/log_job_{job_number:03}.txt"):
        job_number +=1
    Path(f"mdgb/log_job_{job_number:03}.txt").touch()
    
    options.logFile = f"mdgb/log_job_{job_number:03}.txt"
    with Toil(options) as toil:
        if not toil.options.restart:
            config.endstate_files.complex_parameter_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.complex_parameter_filename)))
            config.endstate_files.complex_coordinate_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.complex_coordinate_filename)))
            config.endstate_files.ligand_coordinate_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.ligand_coordinate_filename)))  
            config.endstate_files.ligand_parameter_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.ligand_parameter_filename)))
            config.inputs["min_mdin"] = str(toil.import_file("file://" + os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/templates/min.mdin")))
            
            if not config.ignore_receptor:
                config.endstate_files.receptor_parameter_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.receptor_parameter_filename)))
                config.endstate_files.receptor_coordinate_filename = str(toil.import_file("file://" + os.path.abspath(config.endstate_files.receptor_coordinate_filename)))
            
            if config.endstate_method.endstate_method_type == 'remd':
                for index, (equil_mdin, remd_mdin) in enumerate(zip(config.endstate_method.remd_args.equilibration_replica_mdins, config.endstate_method.remd_args.remd_mdins)):
                    config.endstate_method.remd_args.equilibration_replica_mdins[index] = str(toil.import_file("file://" + os.path.abspath(equil_mdin)))
                    config.endstate_method.remd_args.remd_mdins[index] = str(toil.import_file("file://" + os.path.abspath(remd_mdin)))
            toil.start(Job.wrapJobFn(ddm_workflow, config))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.restraints import write_restraint_forces
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.simulations import Simulation

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184

def compute_mbar(simulation_data: list[pd.DataFrame], temperature:float, matrix_order:CycleSteps, system:str,
                  memory="2G", cores=1, disk="3G"):
    
    
    def create_mbar_format():
        df = pd.concat(simulation_data, axis=0, ignore_index=True)
        print(f"df : {df}")
        
        # df = df["solute"].iloc[0]
        df = df.set_index(
            [
                "solute",
                "parm_state",
                "extdiel",
                "charge",
                "parm_restraints",
                "traj_state",
                "traj_extdiel",
                "traj_charge",
                "traj_restraints",
                "Frames",
            ],
            drop=True,
        )
        df = df[["ENERGY"]]
        df = df.unstack(["parm_state", "extdiel",  "charge", "parm_restraints"])  # type: ignore
        df = df.reset_index(["Frames", "solute"], drop=True)
        states = [_ for _ in zip(*df.columns)][1]
        extdiels = [_ for _ in zip(*df.columns)][2]
        charges = [_ for _ in zip(*df.columns)][3]
        restraints = [_ for _ in zip(*df.columns)][4]

        column_names = [
            (state, extdiel, charge, restraint)
            for state, extdiel, charge, restraint in zip(states, extdiels, charges, restraints)
        ]

        df.columns = column_names  # type: ignore

        # divide by Kcal per Kt
        #kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature
        print(f"Created Unique MBAR dataframe {df.index.unique()}\n")
        return (df / kcals_per_Kt)
    
    kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature
    

    df_mbar = create_mbar_format()
    

    if system == "complex":
        df_mbar = df_mbar[matrix_order.complex_order]
    
    elif system == "ligand":
        df_mbar = df_mbar[matrix_order.ligand_order]
        
    else:
        df_mbar = df_mbar[matrix_order.receptor_order]
    
    
    equil_info = pdmbar.detectEquilibration(df_mbar)

    df_subsampled = pdmbar.subsampleCorrelatedData(df_mbar, equil_info=equil_info)

    return  (pdmbar.mbar(df_subsampled), df_mbar)

    fe, error, mbar = pdmbar.mbar(df_subsampled)

    fe = fe * kcals_per_Kt

    # then multiply by kt
    error = error * kcals_per_Kt


def adpative(job, complex_runner: IntermidateRunner,ligand_runner:IntermidateRunner, receptor_runner:IntermidateRunner,config:Config ):

    #job.log(f"complex_runner: {complex_runner.post_output}")
    cycle_steps = CycleSteps(conformation_forces=config.intermidate_args.exponent_conformational_forces,
                            orientational_forces=config.intermidate_args.exponent_orientational_forces,
                            charges_windows=config.intermidate_args.charges_lambda_window,
                            external_dielectic=config.intermidate_args.gb_extdiel_windows)
    cycle_steps.round(3)
    job.log(f"The start matrix: {cycle_steps.start_restraint_matrix}\n")
    job.log(f"complex steps: \n {cycle_steps.complex_order}")
    results = compute_mbar(simulation_data=complex_runner.post_output, 
                               temperature=config.intermidate_args.temperature, matrix_order=cycle_steps, 
                               system="complex")

    average = overlap_average(results[0][-1].computeOverlap()["matrix"], cycle_steps.start_restraint_matrix)
    job.log(f"Average from overlap {average}")
    
    #job.addChildJobFn(improve_overlap, complex_runner, ligand_runner, receptor_runner, average, config).rv()
    
     
    if good_enough(average=average):
        job.log(f"THE OVERLAP IS ABOVE 0.03")
        ligand_mbar = job.wrapFn(compute_mbar, simulation_data=ligand_runner.post_output, temperature=config.intermidate_args.temperature, 
                                 matrix_order=cycle_steps, system="ligand"
                                )
        receptor_mbar = job.wrapFn(compute_mbar, receptor_runner.post_output, config.intermidate_args.temperature,
                                   matrix_order=cycle_steps, system="receptor"
                                )
        
        return results, job.addChild(ligand_mbar).rv(), job.addChild(receptor_mbar).rv(), config
    
    else:
        job.log(f"POOR OVERLAP BEGING adding windows")
        #improve the method 
        improve_job = job.addChildJobFn(improve_overlap, complex_runner, ligand_runner, receptor_runner, average, config)
        job.addFollowOnJobFn(adpative, improve_job.rv(0), improve_job.rv(1), improve_job.rv(2), improve_job.rv(3))



def overlap_average(overlap_matrix, start):
    
    overlap_neighbors = group_overlap_neighbors(overlap_matrix)
    
    return [(x[0]+x[1])/2 for x in overlap_neighbors[start:]]
       
def good_enough(average):
    """Look through the matrix array for poor overlap 

    Args:
        mbar_results (_type_): _description_
    """
    
    return all([x>0.03 for x in average])

    

def group_overlap_neighbors(matrix):
        
    size = matrix.shape[0] - 1
    
    
    def get_overlap_neighbors(n=0, new=[]):
    
        if n == size:
            return new 
        
        else:
            a = round(matrix[n, n+1],2)
            b = round(matrix[n+1, n],2)
            new.append((a,b))

            
            return get_overlap_neighbors(n+1 ,new=new)
    
    return get_overlap_neighbors()

def improve_overlap(job, comp:IntermidateRunner, ligand:IntermidateRunner, receptor:IntermidateRunner, avg_overlap, config:Config):

    conformational = config.intermidate_args.exponent_conformational_forces.copy()
    orient = config.intermidate_args.exponent_orientational_forces.copy()

    restraints_job = job.addChildJobFn(initilized_jobs)
    job.log(f"Conformational windows: {conformational}\n")
    job.log(f"Orientational windows: {orient}\n")
    for index, a in enumerate(avg_overlap):
        con_window = conformational[index] - 1
        orient_window = orient[index] - 1
        
        if a < 0.03:
            
            #config.lower_bound != None
            if con_window in config.intermidate_args.exponent_conformational_forces:
                job.log(f"INDEX VALUE {index}")
                job.log(f"Divide: {con_window} + {conformational[index]}")
                #bisect 
                con_window = (con_window +  conformational[index])/2
                orient_window = (orient_window + orient[index])/2
                
            job.log(f"ADD WINDOWS\n")
            new_con = np.exp2(con_window)
            new_orient = np.exp2(orient_window)
            job.log(f"Conformational window: {con_window}. np.exp2({con_window}): {new_con}\n")
            job.log(f"Orientational window: {orient_window}\n")
            comp._add_complex_simulation(conformational=con_window, orientational=orient_window, 
                                                mdin=config.inputs["default_mdin"], restraint_file=restraints_job.addChildJobFn(write_restraint_forces, 
                                                                                                    conformational_template=comp.restraints.complex_conformational_restraints,
                                                                                                    orientational_template=comp.restraints.boresch.boresch_template,
                                                                                                    conformational_force=new_con,
                                                                                                    orientational_force=new_orient))
            ligand._add_ligand_simulation(conformational=con_window, mdin=config.inputs["default_mdin"], restraint_file = restraints_job.addChildJobFn(write_restraint_forces,
                                                                                                                                conformational_template=ligand.restraints.ligand_conformational_restraints,
                                                                                                                                conformational_force=new_con))
            receptor._add_receptor_simulation(conformational=con_window, mdin=config.inputs["default_mdin"], restraint_file = restraints_job.addChildJobFn(write_restraint_forces,
                                                                                                                                conformational_template=ligand.restraints.receptor_conformational_restraints,
                                                                                                                                conformational_force=new_con))
            config.intermidate_args.exponent_conformational_forces.append(con_window)
            config.intermidate_args.exponent_orientational_forces.append(orient_window)
    
    config.intermidate_args.exponent_conformational_forces.sort(reverse=True)
    config.intermidate_args.exponent_orientational_forces.sort(reverse=True)
            
    restraints_done = restraints_job.addFollowOnJobFn(initilized_jobs)    
    job.log(f"RUNNER with new restraints window")
    # new_com = comp.new_runner(config, comp.__dict__)
    # job.log(f"new simulation args for restriants RECEPTOR {new_com.simulations[-1].directory_args}")
    # restraints_done.addChild(new_com)
    # restraints_done.addChild(comp.new_runner(config, comp.__dict__))
    #restraints_done.addChild(ligand.new_runner(config, ligand.__dict__))
    #restraints_done.addChild(receptor.new_runner(config, receptor.__dict__))
    
    return (restraints_done.addChild(comp.new_runner(config, comp.__dict__)).rv(), 
            restraints_done.addChild(ligand.new_runner(config, ligand.__dict__)).rv(), 
            restraints_done.addChild(receptor.new_runner(config, receptor.__dict__)).rv(), 
            config
        )


def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return
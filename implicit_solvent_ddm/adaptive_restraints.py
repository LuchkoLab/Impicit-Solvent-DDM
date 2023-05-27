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

def compute_mbar(simulation_data: list[pd.DataFrame], temperature:float, matrix_order:CycleSteps, system:str):
    
    
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
    results, df = compute_mbar(simulation_data=complex_runner.post_output, 
                               temperature=config.intermidate_args.temperature, matrix_order=cycle_steps, 
                               system="complex")

    average = overlap_average(results[-1].computeOverlap()["matrix"], cycle_steps.start_restraint_matrix)
    job.log(f"Average from overlap {average}")
    
    job.addChildJobFn(improve_overlap, complex_runner, ligand_runner, receptor_runner, average, config).rv()
    # if good_enough(average=average):
    #     job.log(f"THE OVERLAP IS ABOVE 0.03")
    #     ligand_mbar = job.wrapFn(compute_mbar, ligand_runner.post_only, config.intermidate_args.temperature)
    #     receptor_mbar = job.wrapFn(compute_mbar, receptor_runner.post_only, config.intermidate_args.temperature)
        
    #     return results, job.addChildJobFn(receptor_mbar), job.addChildJobFn(ligand_mbar)
    
    # else:
    #     job.log(f"POOR OVERLAP BEGING adding windows")
    #     #improve the method 
    #     job.addFollowonJobFn(adpative, job.addChildJobFn(improve_overlap, complex_runner, ligand_runner, receptor_runner, average).rv())



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
    
    
    def newrecur(n=0, new=[]):
    
        if n == size:
            return new 
        
        else:
            a = round(matrix[n, n+1],2)
            b = round(matrix[n+1, n],2)
            new.append((a,b))

            
            return newrecur(n+1 ,new=new)
    
    return newrecur()

def improve_overlap(job, comp:IntermidateRunner, ligand:IntermidateRunner, receptor:IntermidateRunner, avg_overlap, config):

    conformational = config.intermidate_args.exponent_conformational_forces
    orient = config.intermidate_args.exponent_orientational_forces

    for index, a in enumerate(avg_overlap):
        if a < 0.03:
            #config.lower_bound != None
            if False:
                #bisect 
                pass
            else:
                job.log(f"ADD WINDOWS\n")
                new_con = np.exp2(conformational[index] - 1)
                new_orient = np.exp2(orient[index] - 1)
                job.log(f"Conformational window: {new_con}\n")
                job.log(f"Orientational window: {new_orient}\n")
                restraints_job = job.addChildJobFn(initilized_jobs)
                comp._add_complex_simulation(conformational=conformational[index] - 1, orientational=orient[index] - 1, 
                                                 mdin=config.inputs["default_mdin"], restraint_file=restraints_job.addChildJobFn(write_restraint_forces, 
                                                                                                       conformational_template=comp.restraints.complex_conformational_restraints,
                                                                                                       orientational_template=comp.restraints.boresch.boresch_template,
                                                                                                       conformational_force=new_con,
                                                                                                       orientational_force=new_orient))
                ligand._add_ligand_simulation(conformational=new_con, mdin=config.inputs["default_mdin"], restraint_file = restraints_job.addChildJobFn(write_restraint_forces,
                                                                                                                                  conformational_template=ligand.restraints.ligand_conformational_restraints,
                                                                                                                                  conformational_force=new_con))
                receptor._add_receptor_simulation(conformational=new_con, mdin=config.inputs["default_mdin"], restraint_file = restraints_job.addChildJobFn(write_restraint_forces,
                                                                                                                                  conformational_template=ligand.restraints.receptor_conformational_restraints,
                                                                                                                                  conformational_force=new_con))
                config.intermidate_args.exponent_conformational_forces.append(conformational[index] - 1)
                config.intermidate_args.exponent_orientational_forces.append(orient[index] - 1)
                
                restraints_done = restraints_job.addFollowOnJobFn(initilized_jobs)    
    job.log(f"RUNNER with new restraints window")
    new_com = receptor.new_runner(config, comp.__dict__)
    job.log(f"new simulation args for restriants RECEPTOR {new_com.simulations[-1].directory_args}")
    restraints_done.addChild(new_com.simulations[-1])
    #restraints_done.addChild(comp.new_runner(config, comp.__dict__))
    #return (restraints_done.addChild(comp.new_runner(config, comp.__dict__)), restraints_done.addChild(ligand.new_runner(config, ligand.__dict__)), restraints_done.addChild(receptor.new_runner(config, receptor.__dict__)), config)



def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return
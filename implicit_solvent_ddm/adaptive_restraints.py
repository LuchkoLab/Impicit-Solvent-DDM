"""
A collection of functions that performs simple iterative proceess to improve space phase overlap between adjecent states. 
"""

import copy
from typing import Optional

import numpy as np
import pandas as pd

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm.alchemical import alter_topology
from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.matrix_order import CycleSteps
from implicit_solvent_ddm.restraints import write_restraint_forces
from implicit_solvent_ddm.runner import IntermidateRunner
from implicit_solvent_ddm.mdin import generate_extdiel_mdin

AVAGADRO = 6.0221367e23
BOLTZMAN = 1.380658e-23
JOULES_PER_KCAL = 4184


def compute_mbar(
    simulation_data: list[pd.DataFrame],
    temperature: float,
    matrix_order: Optional[CycleSteps],
    system: str,
    memory="2G",
    cores=1,
    disk="3G",
):
    """Execute MBAR analysis.

    Arrange and structure DataFrames to perform MBAR analysis.

    Parameters
    ----------
    simulation_data: list[pd.DataFrame]
        A completed mdout output that contains system information including timestep energies, and temperature.
    temperature: float
        Specified thermostat temperature used in MD simulations.
    matrix_order: CycleSteps
        Arranges the MBAR matrix in chronological order depending on the system.
    system: str
        Denoting the chronogical order of the matrix (i.e. complex, receptor or ligand).

    Returns
    -------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """

    def create_mbar_format():
        df = pd.concat(simulation_data, axis=0, ignore_index=True)

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
        df = df.unstack(["parm_state", "extdiel", "charge", "parm_restraints"])  # type: ignore
        df = df.reset_index(["Frames", "solute"], drop=True)
        states = [_ for _ in zip(*df.columns)][1]
        extdiels = [_ for _ in zip(*df.columns)][2]
        charges = [_ for _ in zip(*df.columns)][3]
        restraints = [_ for _ in zip(*df.columns)][4]

        column_names = [
            (state, extdiel, charge, restraint)
            for state, extdiel, charge, restraint in zip(
                states, extdiels, charges, restraints
            )
        ]

        df.columns = column_names  # type: ignore

        # divide by Kcal per Kt
        # kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature
        print(f"Created Unique MBAR dataframe {df.index.unique()}\n")

        return df / kcals_per_Kt

    kcals_per_Kt = ((BOLTZMAN * (AVAGADRO)) / JOULES_PER_KCAL) * temperature

    df_mbar = create_mbar_format()

    print(f"df.index.unique :\n {df_mbar.index.unique()}\n")
    print(f"df.index.values :\n {df_mbar.index.values}\n")
    print(f"df.columns.unique :\n {df_mbar.columns.unique()}\n")
    print(f"df.columns.values :\n {df_mbar.columns.values}\n")
    # flat bottom to no flat bottom -> EXP()
    if matrix_order is None:
        pass

    elif system == "complex":
        df_mbar = df_mbar[matrix_order.complex_order]

    elif system == "ligand":
        df_mbar = df_mbar[matrix_order.ligand_order]

    else:
        df_mbar = df_mbar[matrix_order.receptor_order]

    equil_info = pdmbar.detect_equilibration(df_mbar)

    df_subsampled = pdmbar.subsample_correlated_data(df_mbar, equil_info=equil_info)

    print("performing MBAR")
    return pdmbar.mbar(df_subsampled), df_mbar

    # return pdmbar.mbar(df_subsampled), df_mbar


def adaptive_lambda_windows(
    job,
    system_runner: IntermidateRunner,
    config: Config,
    system_type: str,
    restraints_scaling: bool = False,
    charge_scaling: bool = False,
    gb_scaling: bool = False,
):
    """
    Simple iterative process to improve poor space phase overlap between restraints and/or ligand charge windows.

    Parameters
    ----------
    job: Toil.job
        The atomic unit of work in a Toil workflow is a Job.
    system_runner: IntermidateRunner
        An system specific runner object to create and inital any new MD runs needed.
    config: Config
        User specified configuration file containing necesssary input information.
    system_type: str
        System type to denote the specific system_runner (i.e. complex, receptor or ligand)
    restraints_scaling: bool
        Whether to perfom adaptive process for restraint windows.
    charge_scaling: bool
        Whether to perform adaptive process for ligand charge scaling.
    gb_scaling: bool
        Whether to perform adaptive procces for scaling GB external dielectric.

    Returns
    -------
    results: tuple[DataFrame, DataFrame, MBAR], pd.DataFrame
        Return values from compute_mbar(*args) function.
    updated_config: Config
        An updated config object with newly added restraint or ligand charge windows.
    """

    updated_config = copy.deepcopy(config)
    # Sort all thermodynamic cycle steps in chronological order
    job.log(f"THE SYSTEM PASSED {system_type}")
    job.log(
        f"conformational exponets: {updated_config.intermidate_args.exponent_conformational_forces}"
    )
    job.log(
        f"orientational exponets: {updated_config.intermidate_args.exponent_orientational_forces}"
    )
    cycle_steps = CycleSteps(
        conformation_forces=updated_config.intermidate_args.exponent_conformational_forces,
        orientational_forces=updated_config.intermidate_args.exponent_orientational_forces,
        charges_windows=updated_config.intermidate_args.charges_lambda_window,
        external_dielectic=updated_config.intermidate_args.gb_extdiel_windows,
    )
    # round all restraint forces values to 3 sig. figs for readable dataframes.
    cycle_steps.round(3)
    job.log(f"THE SYSTEM PASSED {system_type}")
    job.log(f"complex ordered steps: {cycle_steps.complex_order}")
    # Compute MBAR
    results = compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=updated_config.intermidate_args.temperature,
        matrix_order=cycle_steps,
        system=system_type,
    )
    job.log(f"SYSTEM TYPE {system_type}")
    if restraints_scaling:
        job.log("Attempting to improve restraint windows space phase overlap")
        func = improve_restraints_overlap
        job.log(f"set the job function to: {func}")
        matrix_start = cycle_steps.halo_restraint_matrix
        job.log(f"Matrix start: {matrix_start}")
        matrix_end = None
        updated_config.intermidate_args.exponent_conformational_forces.sort(
            reverse=True
        )
        updated_config.intermidate_args.exponent_orientational_forces.sort(reverse=True)
        job.log(
            f"conformational sorted in reverse: {updated_config.intermidate_args.exponent_conformational_forces}"
        )
        job.log(
            f"orientaitonal sorted in reverse: {updated_config.intermidate_args.exponent_orientational_forces}"
        )

        if system_type != "complex":

            matrix_start = 0
            matrix_end = cycle_steps.apo_end_restraint_matrix
            # sort in acending order
            updated_config.intermidate_args.exponent_conformational_forces.sort()

    # Check the overlap for ligand charge scaling
    if charge_scaling:
        job.log("Attempting to improve ligand charge windows space phase overlap")

        func = improve_charge_scaling
        job.log(f"set the job function to: {func}")
        matrix_start = cycle_steps.start_complex_charge_matrix
        job.log(f"Matrix start charge scaling complex: {matrix_start}")
        matrix_end = cycle_steps.halo_restraint_matrix
        # updated_config.intermidate_args.charges_lambda_window.sort()

        if system_type == "ligand":
            matrix_start = cycle_steps.start_ligand_charge_matrix
            matrix_end = None
            # updated_config.intermidate_args.charges_lambda_window.sort(reverse=True)

    # GB external dielectric scaling
    if gb_scaling:
        job.log(
            "Attempting to improve GB external dielectric windows space phase overlap"
        )
        func = improve_gb_dielectric
        job.log(f"set the job function to: {func}")

        matrix_start = cycle_steps.start_gb_extdiel_matrix
        # Once reached the ligand charge scaling stop
        matrix_end = cycle_steps.start_complex_charge_matrix
        job.log(f"GB extdiel matrix_start: {matrix_start}")
        job.log(f"GB extdiel end_start: {matrix_end}")
        updated_config.intermidate_args.gb_extdiel_windows.sort()

    # Get the averaged space phase overlap for specified adaptive step
    # (i.e. ligand charge scaling, restraint scaling or GB scaling)
    averages = overlap_average(
        results[0][-1].compute_overlap()["matrix"],
        matrix_start,
        end=matrix_end,
    )
    job.log(f"The current windows averages: {averages}")

    if good_enough(
        space_phase_overlaps=averages,
        min=updated_config.intermidate_args.min_degree_overlap,
    ):
        job.log(
            f"THE OVERLAP IS ABOVE {updated_config.intermidate_args.min_degree_overlap}"
        )

        return (
            results,
            updated_config,
            system_runner,
        )
    # Not all windows have sufficient overlap
    else:
        job.log("POOR OVERLAP improving restraint windows")
        # improve the overlap for restraint windows

        improve_job = job.addChildJobFn(
            func,
            system_runner,
            averages,
            updated_config,
            system_type,
        )

        # iterative process until all windows reached a sufficient overlap
        return improve_job.addFollowOnJobFn(
            adaptive_lambda_windows,
            improve_job.rv(0),
            improve_job.rv(1),
            system_type=system_type,
            restraints_scaling=restraints_scaling,
            charge_scaling=charge_scaling,
            gb_scaling=gb_scaling,
        ).rv()


def overlap_average(overlap_matrix, start, end=None):
    """
    Compute the average of the degree of space phase overlap between a slice adjacent states.

    Parameters
    ----------
    overlap_matrix: list
        Overlap matrix between the states.
    start: int
        Where the matrix should start reading degree of phase space.
    end: int
        The position to end the matrix.

    Returns
    ------
    A list of averages of the degree of phase space overlap between adjacent states.
    """

    overlap_neighbors = group_overlap_neighbors(overlap_matrix)
    print(f"OVERLAP NEIGH: {overlap_neighbors}")
    restraints_overlap = overlap_neighbors[start:]

    if end is not None:
        restraints_overlap = overlap_neighbors[start:end]

    print(f"RESTRAINT OVERLAPS NUMBERS: {restraints_overlap}")
    return [(x[0] + x[1]) / 2 for x in restraints_overlap]


def good_enough(space_phase_overlaps, min=0.03):
    """Check that all averge degree of overlap are about the minimum criteria.

    Parameters
    ----------
    averages: list
        A list of averages degree of overlap between adjacent stats.
    min: float
        Minimum criteria of degree of overlap percent. Default value = 0.03 (3% overlap)

    Returns
       A boolean wheter all overlap are above the minimum criteria.
    """

    return all([x >= min for x in space_phase_overlaps])


def group_overlap_neighbors(matrix):
    """
    Retireve both the foward and reverse degree of overlap between adjacent states.

    Parameters
    ----------
    matrix: np.ndarray
        Estimated state overlap matrix : O[i,j] is an estimate of the probability of observing a sample from state i in state j

    Returns
    -------
    overlap_neighbors: List[tuple[float, float]]
        Returns a list of tuples of estimated probability of both forward and reverse degree of overlap.
    """
    size = matrix.shape[0] - 1

    def get_overlap_neighbors(n=0, new=[]):
        if n == size:
            return new

        else:
            a = round(matrix[n, n + 1], 2)
            b = round(matrix[n + 1, n], 2)
            new.append((a, b))

            return get_overlap_neighbors(n + 1, new=new)

    return get_overlap_neighbors()


def improve_restraints_overlap(
    job,
    runner: IntermidateRunner,
    avg_overlap,
    config: Config,
    system_type: str,
):
    """
    Improve poor space overlap of two adjacent states via bisection.

    If the user specifed an upper and lower bound (within the configuration file) an biscetion
    will be attempted to improve the overlap between the upper and lower bounds. If only provided
    an upper bound limit than a subtraction of 1 will be computed and biscetion when needed.

    Parameters
    ----------
    job: Toil.job
        The atomic unit of work in a Toil workflow is a Job.
    runner: IntermidateRunner
        An system specific runner object to create and inital any new MD runs needed.
    avg_overlap: list
        A list of averages degree of overlap between adjacent stats.
    config: Config
        User specified configuration file containing necesssary input information.
    system_type: str
        System type to denote the specific system_runner (i.e. complex, receptor or ligand)

    Returns
    -------
    A runner job promise (toil.job.Promise) is essentially a pointer to for the return value that is replaced by the actual return value once it has been evaluated. If any windows were created MD simulation and post-process analysis will be performed and returned.
    config: Config
        An updated configuration file with new inserted states.
    """
    conformational = config.intermidate_args.exponent_conformational_forces.copy()
    orient = config.intermidate_args.exponent_orientational_forces.copy()

    restraints_job = job.addChildJobFn(initilized_jobs)
    job.log(f"Conformational windows: {conformational}\n")
    job.log(f"Orientational windows: {orient}\n")
    job.log(f"Interating over windows {avg_overlap}")

    for index, overlap in enumerate(avg_overlap):
        job.log(f"current window and index: {overlap} {index}")
        if overlap < config.intermidate_args.min_degree_overlap:
            # try to bisect
            try:
                # complex restraints --> releasing restraints moving forward
                new_index = index + 1
                # ligand/receptor endstate->lower_bound so we need know previous overlap -1
                if system_type != "complex":
                    new_index = index - 1

                if new_index < 0:
                    raise IndexError

                con_window = bisect_between(
                    conformational[index], conformational[new_index]
                )
                orient_window = bisect_between(orient[index], orient[new_index])

                job.log(
                    f"No exception was thrown, NEW EXPONENTS con_window and orient_window: {con_window}, {orient_window}"
                )
            except IndexError:
                job.log(f"A IndexError was thrown")
                job.log(f"Subtracting {conformational[index]} - 1")
                job.log(f"Subtracting {orient[index]} - 1")
                con_window = conformational[index] - 1
                orient_window = orient[index] - 1

                job.log(f"con window: {con_window}")
                job.log(f"orient window: {orient_window}")

            job.log(f"ADD WINDOWS\n")
            new_con = np.exp2(con_window)
            new_orient = np.exp2(orient_window)
            job.log(
                f"Conformational FORCE window: {con_window}. np.exp2({con_window}): {new_con}\n"
            )
            job.log(f"Orientational FORCE window: {orient_window}\n")

            if system_type == "complex":
                runner._add_complex_simulation(
                    conformational=con_window,
                    orientational=orient_window,
                    mdin=config.inputs["default_mdin"],
                    restraint_file=restraints_job.addChildJobFn(
                        write_restraint_forces,
                        conformational_template=runner.restraints.complex_conformational_restraints,
                        orientational_template=runner.restraints.boresch.boresch_template,
                        conformational_force=new_con,
                        orientational_force=new_orient,
                    ),
                )

            elif system_type == "ligand":
                runner._add_ligand_simulation(
                    conformational=con_window,
                    mdin=config.inputs["default_mdin"],
                    restraint_file=restraints_job.addChildJobFn(
                        write_restraint_forces,
                        conformational_template=runner.restraints.ligand_conformational_restraints,
                        conformational_force=new_con,
                    ),
                )

            else:
                runner._add_receptor_simulation(
                    conformational=con_window,
                    mdin=config.inputs["default_mdin"],
                    restraint_file=restraints_job.addChildJobFn(
                        write_restraint_forces,
                        conformational_template=runner.restraints.receptor_conformational_restraints,
                        conformational_force=new_con,
                    ),
                )

            config.intermidate_args.exponent_conformational_forces.append(con_window)
            config.intermidate_args.exponent_orientational_forces.append(orient_window)

    if system_type != "complex":
        config.intermidate_args.exponent_conformational_forces.sort()
        config.intermidate_args.exponent_conformational_forces.sort()
    else:
        config.intermidate_args.exponent_conformational_forces.sort(reverse=True)
        config.intermidate_args.exponent_orientational_forces.sort(reverse=True)

    restraints_done = restraints_job.addFollowOnJobFn(initilized_jobs)
    job.log(f"RUNNER with new restraints window")

    return (
        restraints_done.addChild(runner.new_runner(config, runner.__dict__)).rv(),
        config,
    )


def improve_charge_scaling(
    job,
    runner: IntermidateRunner,
    avg_overlap,
    config: Config,
    system_type: str,
):
    """Improve poor space overlap of two adjacent states via bisection.

    If the user specifed an upper and lower bound (within the configuration file) an biscetion
    will be attempted to improve the overlap between the upper and lower bounds. If only provided
    an upper bound limit than a subtraction of 1 will be computed and biscetion when needed.
    This adaptive procedure will be applied towards ligand net charge bisceting between
    1 (ligand fully charge) to 0 (ligand's net charge = 0).
    Args:
        job (_type_): _description_
        runner (IntermidateRunner): _description_
        avg_overlap (_type_): _description_
        config (Config): _description_
        system_type (str): _description_

    Parameters
    ----------
    job: Toil.job
        The atomic unit of work in a Toil workflow is a Job.
    runner: IntermidateRunner
        An system specific runner object to create and inital any new MD runs needed.
    avg_overlap: list
        A list of averages degree of overlap between adjacent stats.
    config: Config
        User specified configuration file containing necesssary input information.
    system_type: str
        System type to denote the specific system_runner (i.e. complex, receptor or ligand)

    Returns
    -------
    A runner job promise (toil.job.Promise) is essentially a pointer to for the return value that is replaced by the actual return value once it has been evaluated. If any windows were created MD simulation and post-process analysis will be performed and returned.
    config: Config
        An updated configuration file with new inserted states.
    """
    charges = config.intermidate_args.charges_lambda_window.copy()

    job.log(f"Interating over windows {avg_overlap}")

    for index, overlap in enumerate(avg_overlap):
        # if sufficient overlap continue iterating
        if overlap > config.intermidate_args.min_degree_overlap:
            continue
        # biscet between adjacent windows with poor overlap
        new_charge = bisect_between(charges[index], charges[index + 1])
        # append to list
        job.log(
            f"Biscent between charges {charges[index], charges[index + 1]} = {new_charge}"
        )
        config.intermidate_args.charges_lambda_window.append(new_charge)
        # check to see if the system passed is an complex
        if system_type == "complex":
            runner._add_complex_simulation(
                conformational=max(
                    config.intermidate_args.exponent_conformational_forces
                ),
                orientational=max(
                    config.intermidate_args.exponent_orientational_forces
                ),
                mdin=config.inputs["default_mdin"],
                restraint_file=runner.restraints.max_complex_restraint,
                charge=new_charge,
                charge_parm=job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.complex_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=new_charge,
                ),
            )
        else:
            # if not a complex then it must be a ligand only system
            runner._add_ligand_simulation(
                conformational=max(
                    config.intermidate_args.exponent_conformational_forces
                ),
                mdin=config.inputs["no_solvent_mdin"],
                restraint_file=runner.restraints.max_ligand_conformational_restraint,
                charge=new_charge,
                charge_parm=job.addChildJobFn(
                    alter_topology,
                    solute_amber_parm=config.endstate_files.ligand_parameter_filename,
                    solute_amber_coordinate=config.endstate_files.ligand_coordinate_filename,
                    ligand_mask=config.amber_masks.ligand_mask,
                    receptor_mask=config.amber_masks.receptor_mask,
                    set_charge=new_charge,
                ),
            )

    return (
        job.addFollowOn(runner.new_runner(config, runner.__dict__)).rv(),
        config,
    )


def improve_gb_dielectric(
    job,
    runner: IntermidateRunner,
    avg_overlap,
    config: Config,
    system_type: str,
):
    """_summary_

    Args:
        job (_type_): _description_
        runner (IntermidateRunner): _description_
        avg_overlap (_type_): _description_
        config (Config): _description_
        system_type (str): _description_
    """
    lower_upper_bound = [0.0, 78.5]

    gb_dielectric = config.intermidate_args.gb_extdiel_windows.copy()
    # add in lower & upper bounds then sort
    gb_dielectric = sorted(list(set(gb_dielectric + lower_upper_bound)))

    job.log(f"Interating over GB windows {avg_overlap}")

    for index, overlap in enumerate(avg_overlap):
        # if sufficient overlap continue iterating
        if overlap > config.intermidate_args.min_degree_overlap:
            continue

        # poor overlap
        else:
            # biscet between upper and lower bound
            new_gb_dielectric = bisect_between(
                gb_dielectric[index], gb_dielectric[index + 1]
            )
        # append new dielectric
        config.intermidate_args.gb_extdiel_windows.append(new_gb_dielectric)

        # create new runner simulation
        runner._add_complex_simulation(
            conformational=max(config.intermidate_args.exponent_conformational_forces),
            orientational=max(config.intermidate_args.exponent_orientational_forces),
            mdin=job.addChildJobFn(
                generate_extdiel_mdin,
                user_mdin_ID=config.intermidate_args.mdin_intermidate_file,
                gb_extdiel=new_gb_dielectric,
            ).rv(),
            restraint_file=runner.restraints.max_complex_restraint,
            charge_parm=job.addChildJobFn(
                alter_topology,
                solute_amber_parm=config.endstate_files.complex_parameter_filename,
                solute_amber_coordinate=config.endstate_files.complex_coordinate_filename,
                ligand_mask=config.amber_masks.ligand_mask,
                receptor_mask=config.amber_masks.receptor_mask,
                set_charge=0.0,
            ),
            gb_extdiel=new_gb_dielectric,
        )
    # sort the new added windows
    config.intermidate_args.gb_extdiel_windows.sort()

    return (
        job.addFollowOn(runner.new_runner(config, runner.__dict__)).rv(),
        config,
    )


def bisect_between(start, end):
    """
    Perform a bisection search between two numbers.

    Parameters
    ----------
        start (float): The start of the interval.
        end (float): The end of the interval.

    Returns
    -------
        float: The midpoint between start and end.
    """
    return (start + end) / 2


def run_exponential_averaging(
    job,
    system_runner: IntermidateRunner,
    temperature: float,
):
    """Execute exponential averaging

    Parameters
    ----------
    system_runner: IntermidateRunner
        A runner class that handles system specific simulations.
    temperature: float
        Specified simulation temperature

    Returns:
    --------
    pdmbar.mbar(df_subsampled): tuple[DataFrame, DataFrame, MBAR]
        DataFrames for the free energies differences (Deltaf_ij), error estimates in free energy difference (dDeltaf_ij), and the pyMBAR object.
    df_mbar: pd.DataFrame
        An formated and chronological arrange DataFrame before any MBAR analysis was performed. (Which can be used to create pdfs of MBAR matrix).
    """
    return compute_mbar(
        simulation_data=system_runner.post_output,
        temperature=temperature,
        matrix_order=None,
        system="free_flat_bottom",
    )


def initilized_jobs(job):
    "Place holder to schedule jobs for MD and post-processing"
    return

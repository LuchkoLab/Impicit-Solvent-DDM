import os
import re
import shutil
import subprocess as sp
import sys
from asyncore import file_dispatcher
from importlib.metadata import files
from logging import setLogRecordFactory
from string import Template
from tempfile import tempdir
from typing import Optional, TypedDict, Union

import pytraj as pt
from pydantic import NumberNotGeError
from toil.common import Toil
from toil.job import FunctionWrappingJob, Job

from implicit_solvent_ddm.config import Config
from implicit_solvent_ddm.get_dirstruct import Dirstruct
from implicit_solvent_ddm.restraints import RestraintMaker

working_directory = os.getcwd()
import logging


class Calculation(Job):
    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        prmtop,
        incrd,
        input_file,
        restraint_file: Union[RestraintMaker, str],
        directory_args: TypedDict,
        dirstruct="dirstruct",
        inptraj=None
    ):
        self.executable = executable
        self.mpi_command = mpi_command
        self.num_cores = num_cores
        self.prmtop = prmtop
        self.incrd = incrd
        self.input_file = input_file
        self.restraint_file = restraint_file
        self.inptraj = inptraj
        self.dirstruct = dirstruct
        self.directory_args = directory_args.copy()
        self.calc_setup = False  # This means that the setup has run successfully
        self.exec_list = [self.mpi_command]
        # self.exec_list = []
        self.read_files = {}
        self._output_directory()

    def _output_directory(self):

        dirs = Dirstruct("mdgb", self.directory_args, dirstruct=self.dirstruct)

        output_dir = os.path.join(
            working_directory, dirs.dirStruct.fromArgs(**dirs.parameters)
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        # self._path2dict(dirs)

    def _setLogging(self):

        file_handler = logging.FileHandler(
            os.path.join(self.output_dir, "simulations.log"), mode="w"
        )
        formatter = logging.Formatter(
            "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s"
        )
        file_handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        return self.logger

    def _mdin_restraint(self, fileStore, mdin):

        # scratch_mdin = fileStore.getLocalTempFile()
        # scratch_restraint = fileStore.getLocalTempFile()

        restraint_file = shutil.copyfile(
            self.read_files["restraint_file"], "restraint.RST"
        )

        with open(mdin) as temp:
            template = Template(temp.read())
        final_template = template.substitute(restraint=restraint_file)

        with open("mdin", "w") as temp_mdin:
            temp_mdin.write(final_template)

        mdin_ID = fileStore.writeGlobalFile("mdin", cleanup=True)
        os.remove("mdin")
        return fileStore.readGlobalFile(mdin_ID)

    @staticmethod
    def export_files(fileStore, output_directory, parameter_files):
        restart_files = []
        traj_files = []
        for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                if name in parameter_files:
                    continue
                # output_file = fileStore.writeGlobalFile(name)
                # fileStore.export_file(output_file,"file://" + os.path.abspath(os.path.join(output_directory, os.path.basename(name))))
                if re.match(r".*.rst7.*", name):
                    output_file = fileStore.writeGlobalFile(name, cleanup=True)
                    restart_files.append(str(output_file))
                elif re.match(r".*.nc.*", name):
                    output_file = fileStore.writeGlobalFile(name, cleanup=True)
                    traj_files.append(str(output_file))
                else:
                    output_file = fileStore.writeGlobalFile(name, cleanup=True)
                fileStore.export_file(
                    output_file,
                    "file://"
                    + os.path.abspath(
                        os.path.join(output_directory, os.path.basename(name))
                    ),
                )

        fileStore.logToMaster(f"the current trajectory files {traj_files}")
        fileStore.logToMaster(f"the restart files: {restart_files}")

        return (restart_files, traj_files)

    def run(self, fileStore):
        """Runs the program. All command-line arguments must be set before
        calling this method. Command-line arguments should be set in setup()
        """
        fileStore.logToMaster(f"the self.directory_args {self.directory_args}")
        # self._output_directory()
        self._setLogging()
        self.logger.info(f"{self.directory_args['runtype']}")

        fileStore.logToMaster(f"directory args {self.directory_args}")
        # If this has not been set up yet
        # then raise a stink
        if not self.calc_setup:
            raise RuntimeError(
                "Cannot run a calculation without calling its"
                + " its setup() function!"
            )
        stdout = sys.stdout
        stderr = sys.stderr
        own_handleo = own_handlee = False
        try:
            process_stdout = open(stdout, "w")
            own_handleo = True
        except TypeError:
            process_stdout = stdout
        try:
            process_stderr = open(stderr, "w")
            own_handlee = True
        except TypeError:
            process_stderr = stderr

        files_in_current_directory = os.listdir(
            f"{os.path.dirname(self.read_files['prmtop'])}"
        )

        fileStore.logToMaster(f"file in current directory {files_in_current_directory}")
        fileStore.logToMaster(f"exec_list : {self.exec_list}")
        self.logger.info(
            f"The files in the current working directory: {files_in_current_directory}"
        )
        self.logger.info(f"executable command {self.exec_list}")

        # amber_output = sp.Popen(self.exec_list, stdout=sp.PIPE, stderr=sp.PIPE)
        amber_output = sp.run(self.exec_list, stdout=sp.PIPE, stderr=sp.PIPE)

        amber_stdout = amber_output.stdout.decode("utf-8")
        amber_stderr = amber_output.stderr.decode("utf-8")
        # amber_stdout = amber_output.stdout.read().splitlines()
        # amber_stderr = amber_output.stderr.read().splitlines()

        self.logger.error(f"AMBER stdout: {amber_stdout}")
        self.logger.error(f"AMBER stderr: {amber_stderr}")

        fileStore.logToMaster(f"amber_stdout: {amber_stdout}")
        fileStore.logToMaster(f"amber_stderr: {amber_stderr}")

        # export parameter files
        fileStore.export_file(
            self.read_files["prmtop"],
            "file://"
            + os.path.abspath(
                os.path.join(
                    self.output_dir, os.path.basename(self.read_files["prmtop"])
                )
            ),
        )
        fileStore.export_file(
            self.read_files["restraint_file"],
            "file://"
            + os.path.abspath(
                os.path.join(
                    self.output_dir, os.path.basename(self.read_files["restraint_file"])
                )
            ),
        )

        restart_ID, trajectory_ID = self.export_files(
            fileStore, self.output_dir, files_in_current_directory
        )

        return (restart_ID, trajectory_ID)


class Simulation(Calculation):
    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        prmtop,
        incrd,
        input_file,
        restraint_file: Union[RestraintMaker, str],
        directory_args,
        conformational_force=None, 
        orientational_force=None, 
        dirstruct="dirstruct",
        inptraj=None,
        restraint_key=None,
        memory: Optional[Union[int, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ):
        Job.__init__(self, memory=memory, cores=num_cores, disk=disk)
        Calculation.__init__(
            self,
            executable,
            mpi_command,
            num_cores,
            prmtop,
            incrd,
            input_file,
            restraint_file,
            directory_args,
            dirstruct=dirstruct,
            inptraj=inptraj,
        )
        self.restraint_key = restraint_key 
    
    def setup(self):
        """
        Sets up the command-line arguments. Sander requires a unique restrt file
        for the MPI version (since one is *always* written and you don't want 2
        threads fighting to write the same dumb file)
        """

        if self.num_cores == 1 or self.mpi_command == None:
            self.exec_list.pop(0)
            self.exec_list.append(re.sub(r"\..*", "", self.executable))
        else:
            self.exec_list.extend(("-n", str(self.num_cores)))
            self.exec_list.append(self.executable)

        solu = re.sub(r"\..*", "", os.path.basename(str(self.prmtop)))
        restart_filename = f"{solu}_{self.directory_args['filename']}_restrt"
        trajector_filename = f"{solu}_{self.directory_args['filename']}_traj"
        self.exec_list.append("-O")  # overwrite flag
        self.exec_list.extend(("-i", self.read_files["mdin"]))  # input file flag
        self.exec_list.extend(("-p", self.read_files["prmtop"]))  # prmtop flag
        self.exec_list.extend(("-c", self.read_files["incrd"]))  # input coordinate flag
        self.exec_list.extend(("-r", f"{restart_filename}.rst7"))
        self.exec_list.extend(("-x", f"{trajector_filename}.nc"))
        self.exec_list.extend(("-o", "mdout"))  # output file flag
        if self.inptraj is not None:
            self.exec_list.extend(
                ("-y", self.read_files["inptraj"])
            )  # input trajectory flag

        self.calc_setup = True

    def run(self, fileStore):

        tempDir = fileStore.getLocalTempDir()
        """
        Import all command line arguments into the temporary working directory 
        """

        self.read_files["prmtop"] = fileStore.readGlobalFile(
            self.prmtop, userPath=os.path.join(tempDir, os.path.basename(self.prmtop))
        )
        self.read_files["incrd"] = fileStore.readGlobalFile(
            self.incrd, userPath=os.path.join(tempDir, os.path.basename(self.incrd))
        )
        self.read_files["input_file"] = fileStore.readGlobalFile(
            self.input_file,
            userPath=os.path.join(tempDir, os.path.basename(self.input_file)),
        )
        if self.inptraj is not None:
            self.read_files["inptraj"] = fileStore.readGlobalFile(
                self.inptraj[0],
                userPath=os.path.join(tempDir, os.path.basename(self.inptraj[0])),
            )
        if isinstance(self.restraint_file, RestraintMaker):
            fileStore.logToMaster(f"RESTRAINT {self.restraint_file}")
            fileStore.logToMaster(f"RESTRAINT {self.restraint_file.restraints}")
            self.read_files["restraint_file"] = fileStore.readGlobalFile(
                self.restraint_file.restraints[self.restraint_key],
                userPath=os.path.join(tempDir, os.path.basename(self.restraint_file.restraints[self.restraint_key])), 
            )
        else:
            self.read_files["restraint_file"] = fileStore.readGlobalFile(
                self.restraint_file,
                userPath=os.path.join(tempDir, os.path.basename(self.restraint_file)),
            )
        self.read_files["mdin"] = Calculation._mdin_restraint(
            self, fileStore, self.read_files["input_file"]
        )

        self.setup()

        output = Calculation.run(self, fileStore)

        return output


class REMDSimulation(Calculation):
    def __init__(
        self,
        executable,
        mpi_command,
        num_cores,
        prmtop,
        incrd,
        input_file,
        restraint_file,
        runtype,
        ngroups,
        directory_args,
        dirstruct="dirstruct",
        inptraj=None,
        memory: Optional[Union[int, str]] = None,
        disk: Optional[Union[int, str]] = None,
        preemptable: Optional[Union[bool, int, str]] = None,
        unitName: Optional[str] = "",
        checkpoint: Optional[bool] = False,
        displayName: Optional[str] = "",
        descriptionClass: Optional[str] = None,
    ):
        Job.__init__(self, memory=memory, cores=num_cores, disk=disk)
        Calculation.__init__(
            self,
            executable,
            mpi_command,
            num_cores,
            prmtop,
            incrd,
            input_file,
            restraint_file,
            directory_args,
            dirstruct=dirstruct,
            inptraj=None,
        )
        self.runtype = runtype
        self.ng = ngroups
        self.nthreads = num_cores

    def _setup(self):
        """
        Sets up the REMD calculation. All it has to do is fill in the
        necessary command-line arguments
        """

        self.exec_list.extend(("-n", str(self.num_cores)))
        self.exec_list.append(self.executable)
        self.exec_list.extend(("-ng", str(self.ng)))
        self.exec_list.extend(("-groupfile", self.read_files["groupfile"]))

        self.calc_setup = True

    def _groupfile(self, fileStore):

        scratch_file = fileStore.getLocalTempFile()
        fileStore.logToMaster(f"self.input_file {self.input_file}")
        # create groupfile mdin
        with open(scratch_file, "w") as group:
            for count, mdin in enumerate(self.input_file):
                fileStore.logToMaster(f"mdin for remd {mdin}")
                read_mdin = fileStore.readGlobalFile(
                    mdin, userPath=os.path.join(self.tempDir, os.path.basename(mdin))
                )
                local_mdin = Calculation._mdin_restraint(self, fileStore, read_mdin)
                fileStore.logToMaster(f"local mdin {local_mdin}")
                solu = re.sub(r"\..*", "", os.path.basename(self.prmtop))

                if self.runtype == "equil":
                    group.write(
                        f"""-O -rem 0 -i {local_mdin} 
                    -p {self.read_files["prmtop"]} -c {self.read_files["incrd"]} 
                    -o equilibrate.mdout.{count:03} -inf equilibrate.mdinfo.{count:03}
                    -r {solu}_equilibrate.rst7.{count:03} -x {solu}_equilibrate.nc.{count:03}""".replace(
                            "\n", ""
                        )
                        + "\n"
                    )

                elif self.runtype == "remd":
                    single_coordinate = [
                        coordinate
                        for coordinate in self.incrd
                        if re.search(rf".*.rst7.{count:03}", coordinate)
                    ]
                    read_coordinate = fileStore.readGlobalFile(
                        single_coordinate[0],
                        userPath=os.path.join(
                            self.tempDir, os.path.basename(single_coordinate[0])
                        ),
                    )
                    group.write(
                        f"""-O -rem 1 -remlog rem.log
                        -i {local_mdin} -p {self.read_files["prmtop"]} 
                        -c {read_coordinate} -o remd.mdout.rep.{count:03} 
                        -r remd.rst7.{count:03} -x remd.nc.{count:03} 
                        -inf remd.mdinfo.{count:03}""".replace(
                            "\n", ""
                        )
                        + "\n"
                    )

            groupfile_ID = fileStore.writeGlobalFile(scratch_file)

        self.read_files["groupfile"] = fileStore.readGlobalFile(groupfile_ID)

    def run(self, fileStore):

        tempDir = self.tempDir
        fileStore.logToMaster(f"self.incrd is {self.incrd}")
        # read in parameter files
        self.read_files["prmtop"] = fileStore.readGlobalFile(
            self.prmtop, userPath=os.path.join(tempDir, os.path.basename(self.prmtop))
        )
        self.read_files["restraint_file"] = fileStore.readGlobalFile(
            self.restraint_file,
            userPath=os.path.join(tempDir, os.path.basename(self.restraint_file)),
        )
        # The case in running REMD possible many equilibration restart files
        if len(self.incrd) == 1:
            self.read_files["incrd"] = fileStore.readGlobalFile(
                self.incrd[0],
                userPath=os.path.join(tempDir, os.path.basename(self.incrd[0])),
            )
        if self.inptraj is not None:
            self.read_files["inptraj"] = fileStore.readGlobalFile(
                self.inptraj,
                userPath=os.path.join(tempDir, os.path.basename(self.inptraj)),
            )

        self._groupfile(fileStore)

        self._setup()
        return Calculation.run(self, fileStore)


class ExtractTrajectories(Job):
    def __init__(self, solute_topology, trajectory_files, target_temp=0.0):
        Job.__init__(self, memory="2G", cores=1, disk="3G")
        self.solute_topology = solute_topology
        self.trajectory_files = trajectory_files
        self.target_temp = target_temp
        self.read_trajs = []

    def run(self, fileStore):
        temp_dir = fileStore.getLocalTempDir()
        self.read_solute = fileStore.readGlobalFile(
            self.solute_topology,
            userPath=os.path.join(temp_dir, os.path.basename(self.solute_topology)),
        )

        # extract target temperture in REMD trajectories
        if self.target_temp != 0.0:

            for traj_file in self.trajectory_files:
                self.read_trajs.append(
                    fileStore.readGlobalFile(
                        traj_file,
                        userPath=os.path.join(temp_dir, os.path.basename(traj_file)),
                    )
                )

            write_bash_script = fileStore.writeGlobalFile(self.bash_script)
            read_bash_script = fileStore.readGlobalFile(
                write_bash_script,
                userPath=os.path.join(temp_dir, os.path.basename(write_bash_script)),
            )
            fileStore.logToMaster(f"read bash {read_bash_script}")
            # extract trajectories at target temperature
            # extract_target_temp_traj = self.run_bash(read_bash_script, fileStore)
            solute_traj = self.run_bash(read_bash_script, fileStore)
            read_target_traj = fileStore.readGlobalFile(
                solute_traj,
                userPath=os.path.join(temp_dir, os.path.basename(solute_traj)),
            )
        # user provided there one endstate trajectory
        else:
            read_target_traj = fileStore.readGlobalFile(
                self.trajectory_files,
                userPath=os.path.join(
                    temp_dir, os.path.basename(self.trajectory_files)
                ),
            )
            solute_traj = self.trajectory_files

        solu = re.sub(r"\..*", "", os.path.basename(self.read_solute))
        lastframe = f"{solu}_{self.target_temp}K_lastframe.ncrst"
        lastframe_rst7 = f"{solu}_{self.target_temp}K_lastframe.rst7"

        # get the last frame at the target temperature
        sp.run(
            [
                "cpptraj",
                "-p",
                self.read_solute,
                "-y",
                read_target_traj,
                "-ya",
                "lastframe",
                "-x",
                lastframe,
            ]
        )
        sp.run(
            ["cpptraj", "-p", self.read_solute, "-y", lastframe, "-x", lastframe_rst7],
            capture_output=True,
        )
        sp.run(
            ["cpptraj", "-p", self.read_solute, "-y", lastframe_rst7, "-x", lastframe],
            capture_output=True,
        )

        final_ncrst_frame = fileStore.writeGlobalFile(lastframe)
        final_rst7_frame = fileStore.writeGlobalFile(lastframe_rst7)

        output_path = os.path.join(working_directory, "mdgb/temp_extract_trajectories")
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        fileStore.export_file(
            solute_traj,
            "file://"
            + os.path.abspath(os.path.join(output_path, os.path.basename(solute_traj))),
        )
        fileStore.export_file(
            final_ncrst_frame,
            "file://"
            + os.path.abspath(
                os.path.join(output_path, os.path.basename(final_ncrst_frame))
            ),
        )
        fileStore.export_file(
            final_rst7_frame,
            "file://"
            + os.path.abspath(
                os.path.join(output_path, os.path.basename(final_rst7_frame))
            ),
        )

        return (solute_traj, final_ncrst_frame, final_rst7_frame)

    def run_bash(self, bash_script, fileStore):
        current_files = os.listdir()
        output = sp.run(["bash", bash_script], capture_output=True)
        for file in os.listdir():
            if file not in current_files:
                output_file = fileStore.writeGlobalFile(file)

        return output_file

    @property
    def bash_script(self):
        # initial starting frame trajectory_name.nc.000
        initial_coordinate = list(
            filter(
                lambda coordinate: re.match(r".*\.nc.000", coordinate), self.read_trajs
            )
        )

        bash_script = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/templates/cpptraj_remd.sh"
        )
        with open(bash_script) as t:
            template = Template(t.read())

        solu = re.sub(r"\..*", "", os.path.basename(self.solute_topology))
        output_trajectory_filename = f"{solu}_{self.target_temp}K"

        final_template = template.substitute(
            solute=self.read_solute,
            trajectory=initial_coordinate[0],
            target_temperature=self.target_temp,
            temperature_traj=output_trajectory_filename,
        )
        with open(f"{solu}_cpptraj_extract_{self.target_temp}K.x", "w") as output:
            output.write(final_template)

        return os.path.abspath(f"{solu}_cpptraj_extract_{self.target_temp}K.x")


def write_mdin(job, mdin_type):

    tempdir = job.fileStore.getLocalTempDir()

    mdin_path = (
        "/home/ayoub/nas0/Impicit-Solvent-DDM/implicit_solvent_ddm/templates/min.mdin"
    )

    mdin_import = job.fileStore.import_file("file://" + mdin_path)

    return mdin_import


def write_empty_restraint(job) -> str:

    scratch_file = job.fileStore.getLocalTempFile()

    with open(scratch_file, "w") as rest:
        rest.write("")

    return job.fileStore.writeGlobalFile(scratch_file)


def workflow(job: FunctionWrappingJob, parm_file, coord_files):

    # mdin = job.addChildJobFn(write_mdin, 'min')
    # restriant_file = job.addFollowOnJobFn(write_empty_restraint)
    # return mdin.rv()
    # output = restriant_file.addFollowOn(Simulation(executable='sander', mpi_command='srun', prmtop=parm_file, incrd=coord_file,
    #                                                input_file=mdin.rv(), num_cores=1, restraint_file=restriant_file.rv(),
    #                                                directory_args={"solute": parm_file, "state_label": 5,}))
    output = job.addChild(ExtractTrajectories(parm_file, coord_files[0]))


if __name__ == "__main__":

    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.logLevel = "INFO"
    options.clean = "always"
    ligand_trajs = []
    # cb7 =os.path.abspath("structs/complex/cb7-mol01.parm7")
    # traj = pt.load("inputs/M01_000_minimization.rst7", "inputs/M01_000.parm7")
    with Toil(options) as toil:
        ligand_prmtop = toil.import_file(
            "file://"
            + os.path.abspath(
                "/nas0/ayoub/Impicit-Solvent-DDM/output_directory/inputs/M01_000.parm7"
            )
        )
        ligand_trajs.append(
            toil.import_file(
                "file://"
                + os.path.abspath(
                    "/nas0/ayoub/Impicit-Solvent-DDM/success_postprocess/mdgb/remd/M01_000/remd.nc.000"
                )
            )
        )
        ligand_trajs.append(
            toil.import_file(
                "file://"
                + os.path.abspath(
                    "/nas0/ayoub/Impicit-Solvent-DDM/success_postprocess/mdgb/remd/M01_000/remd.nc.001"
                )
            )
        )
        ligand_trajs.append(
            toil.import_file(
                "file://"
                + os.path.abspath(
                    "/nas0/ayoub/Impicit-Solvent-DDM/success_postprocess/mdgb/remd/M01_000/remd.nc.002"
                )
            )
        )
        ligand_trajs.append(
            toil.import_file(
                "file://"
                + os.path.abspath(
                    "/nas0/ayoub/Impicit-Solvent-DDM/success_postprocess/mdgb/remd/M01_000/remd.nc.003"
                )
            )
        )
        print(toil.start(Job.wrapJobFn(workflow, ligand_prmtop, ligand_trajs)))

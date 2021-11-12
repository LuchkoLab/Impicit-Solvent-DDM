import os 
import logging
import argparse
import sys
import implicit_solvent_ddm.run_dirstruct as run_dirstruct 
import implicit_solvent_ddm.preferences as preferences
import implicit_solvent_ddm.directorystructure as directorystructure 


class Dirstruct():
    '''
    This class provides basic functionality to create directory structure. 
    Utilizes peferences files .json to construct 
    
    Customization
    =============

    Customization is achieved through JSON format preference
    files. Dirstruct looks for preference files in three places:
    
        1. ``implicit_solvent_ddm/templates``,
        2. ``$HOME``, and
        3. ``./``

    It is mandatory to have preference files in ``implicit_solvent_ddm/templates/``.  Other
    locations are optional.  Dirstruct first looks for preferance files in
    the order listed above. Global preference files, ``simrun.json``,
    are searched for first in all locations and then preference files
    for this specific instance, ``<name>.json``.  Optional files have a
    ``.`` prepended.  I.e., they are hidden files.  It then looks in the
    current working directory for ``.<name>.json`` and appends and
    overwrites the default preferences. 
    '''
    def __init__(self, runtype, description=None):
        self._setLogging()
        self.runtype = runtype
        self.description = description
        self._getprefs()

    def _getprefs(self):
        """Find and load the default and local preferences.

        Local preferences are optional and are located in the current
        working directory.

        Default ones are not optional and are located relative to the
        executable.

        Optional preferences are first looked for in the $HOME
        directory and then in the calling directory.

        Global preferences are named 'simrun.json' and are read for
        all simrun executables. Preferences for a specific executable
        are named self.runtype+".json".  Optional preference files are
        preceded with a '.' (hidden files).

        Raises
            RunTimeError - if default not found.
        """
        self.prefs=preferences.Preferences()
        #where to look
        defaultPath =os.path.abspath(os.path.dirname(
                os.path.realpath(__file__)) +'/templates/')
        extraPaths = [os.environ['HOME'],os.getcwd()]
        
        #first look for the global simrun prefs and then those
        #specific to this program
        for rootname in ["simrun",self.runtype]:
            #mandatory files
            prefsDefault=defaultPath+"/"+rootname+".json"
            self.log.debug("Looking for preference file : "+prefsDefault)
            if(os.path.exists(prefsDefault)):
                self.prefs.addFile(prefsDefault)
            else:
                raise RuntimeError("Default preferences not found. "
                                   +"Expected to file at '"+prefsDefault)
            #optional files
            for path in extraPaths:
                prefsExtra=path+"/."+rootname+".json"
                self.log.debug("Looking for preference file : "+prefsExtra)
                if(os.path.isfile(prefsExtra)):
                    self.prefs.addFile(prefsExtra)
                    
    def getRun(self,argset,dirstruct="dirstruct"):
        '''
        Returns a Run object for a given set of arguments.

        Args:
            argset: dict
                a dictionary of key-value pairs.  Typically generated from Arguments.generateSets()
            dirstruct: str
                (optional) name of the dirstruct defined in the preferences file
        
        Returns:
            a Run object
        '''
        return run_dirstruct.RunDirstruct(self.runtype,argset,self.prefs,dirstruct)

    def getDirectoryStructure(self,dirstruct="dirstruct"):
        '''Returns a DirectoryStructure object with the current preferences
        and arguments.  This useful, for example, to get all
        directories consistent with the command line options.

        Args:
            dirstruct: str
                (optional) name of the dirstruct defined in the preferences file

        Returns:
            a DirectoryStructure

        '''
        return directorystructure.DirectoryStructure(self.prefs,self.args,dirstruct)
    
    def _setLogging(self):
        """
        Sets up logging for run logs and the global logging.
        Side Effects:
            sets the logging level for individual run logs and global
            logging.  May also create a global log output file
        """
        #root level logger
        rootlog = logging.getLogger()
        rootlog.setLevel(logging.WARN)

        #formatter for all handlers defined here
        formatter = logging.Formatter(
            '%(levelname)s::%(name)s:: %(message)s')

        #global stderr handler.  Only does warnings unless stderr
        #debugging requested on the command line
        stderrHandler = logging.StreamHandler()
        stderrHandler.setLevel(logging.WARN)
        stderrHandler.setFormatter(formatter)
        rootlog.addHandler(stderrHandler)

        #set the default level for run logs.
        logging.getLogger("simrun.run").setLevel(logging.INFO)

        #this logger
        self.log=logging.getLogger(__name__)

        #grab the debug flag before anything else so we can debug
        #templating etc.
        argparser = argparse.ArgumentParser(add_help=False)
        argparser.add_argument("--debug",nargs="*",
                               choices = ["file","stderr","stdout","all"])
        args,unknown= argparser.parse_known_args()
        args=vars(args)

        #remove debug from arguments.  It causes issues iterating
        #argument sets and isn't necessary for the main program
        sys.argv=[sys.argv[0]]+unknown

        #check options
        if "debug" in args:
            if args["debug"] is not None:
                #turn on debugging for run logs.
                logging.getLogger("simrun.run").setLevel(logging.DEBUG)
                if len(args["debug"]) > 0:
                    if "file" in args["debug"]:
                        fh = logging.FileHandler("debug.log","w")
                        fh.setLevel(logging.DEBUG)
                        fh.setFormatter(formatter)
                        rootlog.addHandler(fh)
                    if "stderr" in args["debug"]:
                        stderrHandler.setLevel(logging.DEBUG)
                    if "stdout" in args["debug"]:
                        stdoutHandler = logging.StreamHandler(sys.stdout)
                        stdoutHandler.setLevel(logging.DEBUG)
                        stdoutHandler.setFormatter(formatter)
                        rootlog.addHandler(stdoutHandler)
                    #check "all" last so debug line gets output
                    if "all" in args["debug"]:
                        rootlog.setLevel(logging.DEBUG)
                        rootlog.debug("Turned on global debugging.")
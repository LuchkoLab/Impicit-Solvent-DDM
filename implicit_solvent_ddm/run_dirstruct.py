import os 
import errno
import copy
#local imports
import implicit_solvent_ddm.preferences as preferences 
import implicit_solvent_ddm.directorystructure as directorystructure 

class RunDirstruct():
    """
    This class is the primary interface for using templates to create
    directories. 
    """
    def __init__(self, runtype, parameters, prefs, dirstruct="dirstruct"):
        self.runtype = runtype
        self.parameters= copy.deepcopy(parameters)
        self.prefs= prefs.copy() 
        self.dirStruct = directorystructure.DirectoryStructure(self.prefs, self.parameters, dirstruct)
        self._createPath()
    #
    def _createPath(self):
        """
        Creates the path and makes the directory that will be the
        working directory of the simulation
        Side Effects:
           Creates a new directory
        """
        self.path = self.dirStruct.fromArgs(**self.parameters)
        self.pathback = os.path.relpath(os.getcwd(),self.path)
        self.pathprev=None
        #self.log.debug("Attempting to create a director for the previous run")
        lastDir = self.prefs[self.dirStruct.key][-1]
        # print('lastDir', lastDir)
        # print('LastDir["variable"]', lastDir["variable"])
        
        if "variable" in lastDir and lastDir["variable"] == "runnumber":
            self.runnumber = int(os.path.basename(self.path))
            # print("Current runnumber is "+str(self.runnumber))
            if self.runnumber !=0:
                self.pathprev=os.path.relpath(
                    # replace the last directory in path with a number one less
                    os.path.join(os.path.dirname(self.path),
                                 ("{0:"+lastDir["format"]+"}").format(self.runnumber-1)),
                    # relative to the current path
                    self.path)
            elif "startdir" in lastDir:
                # get the user defined 'startdir' and make sure it is
                # relative to the current path
                self.pathprev=os.path.relpath(
                    os.path.join(self.path,lastDir["startdir"]),self.path)
            # print("Path to the previous run is "+self.pathprev)
        
        elif "variable" in lastDir and lastDir["variable"] == "traj_number":
            traj_number = int(os.path.basename(self.path))
            print('type(traj_number)',type(traj_number))
            print('type(traj_number)', traj_number)
            print('lastDir["format"]',lastDir["format"])
            if traj_number !=0:
                self.pathprev=os.path.relpath(                                                                                       
                    # replace the last directory in path with a number one less                                                      
                    os.path.join(os.path.dirname(self.path),                                                                         
                                 ("{0:"+"03d"+"}").format(traj_number-1)),
                    # relative to the current path                                                                                   
                    self.path)
            elif "startdir" in lastDir:                                                                                              
                # get the user defined 'startdir' and make sure it is                                                                
                # relative to the current path                                                                                       
                self.pathprev=os.path.relpath(                                                                                       
                    os.path.join(self.path,lastDir["startdir"]),self.path)                                                           
            # print("Path to the previous run is "+self.pathprev) 
        
        else:
            self.pathprev=None
        try:
            os.makedirs(self.path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(self.path):
                pass
            else: 
                print("ERROR: could not create "+self.path)
                raise


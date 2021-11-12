import os, os.path
import re
import logging
import string
import pprint

def reverse_readline(filename, buf_size=8192):
    """a generator that returns the lines of a file in reverse order
    From https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python/23646049#23646049
    """
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if len(lines[index]):
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment
            
class DirectoryStructure(object):
    """
    Generates directory paths based off of preferences templates and
    user command line arguments.
    """
    
    #There is no equivalent to scanf in Python so we use the table
    #from the Re manual page
    scanfRe = {"c":	".",
               "d":	"[-+]?\d+",
               "e": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
               "E": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
               "f": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
               "g": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
               "i":	"[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)",
               "o":	"[-+]?[0-7]+",
               "s":	"\S+",
               "u":	"\d+",
               "x": "[-+]?(0[xX])?[\dA-Fa-f]+", 
               "X": "[-+]?(0[xX])?[\dA-Fa-f]+"}

    def __init__(self,prefs,args,dirstruct="dirstruct"):
        """
        Initialize.

        Args:
            prefs - Preferences object or just the dictionary.
            args - Args object or just the dictionary.
            dirstruct - (optional) keyname for directory structure array
        """
        self.prefs=prefs
        self.args=args
        self.log = logging.getLogger(__name__)
        self.key = dirstruct
        self.log.info("Using directory structure : "+dirstruct)

    def variableNames(self,includeStatic=False):
        """Returns ordered list of variable name handles for each level of
        directory structure.
        Args:
            includeStatic - include static directories in output
        Returns:
            ordered list of names
        """
        names=[]
        for param in self.prefs[self.key]:
            if "variable" in param:
                names.append(param["variable"])
            elif includeStatic and "static" in param:
                names.append(param["static"])
        return names

    def getDirs(self,runnumbers=False):
        """
        Returns a list of all directories compatible with the preferences
        and arguments.  If "runnumber" is part of "dirstruct", the
        next run number not yet created is returned.

        Args:
            runnumbers: Boolean
                If True, include valid runnumber directories up to the
                last valid runnumber on disk.
                If False, return the next runnumber.

        Returns:
            a list of directory paths

        """
        dirs=[]
        for d in self:
            if runnumbers:
                lastdir =self.prefs[self.key][-1]
                if "variable" in lastdir and lastdir["variable"]=="runnumber":
                    head,tail = os.path.split(d)
                    for i in range(int(tail)):
                        dirs.append(
                            os.path.join(
                                head,
                                ("{0:"+lastdir["format"]+"}").format(i)))
            else:
                dirs.append(d)
        return dirs

    def fromArgs(self,returnas=str,includeStatic=True,**kwargs):
        """
        Returns a directory based on the template using the specific
        keyword-value pairs provide. Only those keywords in the
        template are used, if the keyword is missing, an exception is
        thrown
        Args:
           returnas - return as a str, list or dict
           includeStatic - include static directories in output
           **kwargs - list of keyword-value pairs
        Returns:
           a directory in the form of a string
        """
        self.log.debug("Creating directory from arguments.")
        directory=[]
        #regular directories from the user options
        value=""
        for i,param in enumerate(self.prefs[self.key]):
            self.log.debug("Appending directory : "+str(param))
            if "static" in param: 
                self.log.debug("Is static")
                if includeStatic:
                    value=param["static"]
                else:
                    self.log.debug("Skipping static")
                    continue
            elif param["variable"] == "runnumber":
                break
            else:
                value = kwargs[param["variable"]]
                self.log.debug("Is a variable with value : "+
                               str(value)+" "+str(type(value)))
            value = self._cleanupDirname(value,param)
            directory.append(
                ("{0:"+param["format"]+"}").format(value))
            self.log.debug("Growing directory to : "+str(directory))

        #special case of "runnumber"
        lastDir = self.prefs[self.key][-1]
        if "variable" in lastDir and lastDir["variable"] == "runnumber":
            self.log.debug("Last dir is runnumber")
            try:
                nextRun=self._getNextRun(os.path.join(*directory))
            except RuntimeError as e:
                if "startdir" not in self.prefs[self.key][-1]:
                    nextRun = 0
                else:
                    raise e
            directory.append(
                ("{0:"+lastDir["format"]+"}").format(nextRun))
            self.log.debug("Growing directory to : "+str(directory))

        if returnas is dict:
            return dict(zip(self.variableNames(includeStatic),directory))
        if returnas is list:
            return directory
        return os.path.join(*directory)

    def fromList(self,values,returnas=str,includeStatic=True):
        """
        Returns a directory based on the template using an ordered
        list of values.
        Args:
           values - list of values
           returnas - return as a str, list or dict
           includeStatic - include static directories in output
        Returns:
           a directory in the form of a string or dictionary
        """
        self.log.debug("Creating directory from list of values.")
        directory=[]
        
        #regular directories from the user options
        ivalue=0
        for i,param in enumerate(self.prefs[self.key]):
            self.log.debug("Appending directory : "+str(i))
            if "static" in param: 
                self.log.debug("Is static")
                if includeStatic:
                    value=param["static"]
                else:
                    self.log.debug("Skipping static")
                    continue
            elif param["variable"] == "runnumber":
                break
            else:
                value = values[ivalue]
                ivalue+=1
                self.log.debug("Is a variable with value : "+
                               str(value))
            value = self._cleanupDirname(value,param)
            try:
                directory.append(
                    ("{0:"+param["format"]+"}").format(value))
            except ValueError as e:
                self.log.error("'"+str(value)+"' does not match format '"+
                               param["format"]+"'")
                raise e
            self.log.debug("Growing directory to : "+str(directory))

        #special case of "runnumber"
        lastDir = self.prefs[self.key][-1]
        if "variable" in lastDir and lastDir["variable"] == "runnumber":
            self.log.debug("Last dir is runnumber")
            nextRun=self._getNextRun(os.path.join(*directory))
            directory.append(
                ("{0:"+lastDir["format"]+"}").format(nextRun))
            self.log.debug("Growing directory to : "+str(directory))

        if returnas is dict:
            return dict(zip(self.variableNames(includeStatic),directory))
        if returnas is list:
            return directory
        return os.path.join(*directory)

        

    def fromPath2List(self,path):
        """Returns values from path corresponding to the defined directory
        structure as an ordered list. Values in the path will be
        converted to their appropriate type based on their format
        specification in the dirstruct.
        Args:
            trialpath - path to parse
        Returns:
            list of variable values.  'static' names are
            also included.
        Raises:
            ValueError if the path does not conform to the defined
            directory structure.
        """
        values=[]
        subpath=path
        for param in reversed(self.prefs[self.key]):
            subpath,top = os.path.split(subpath)
            if "variable" in param:
                if re.match(self.scanfRe[param["format"][-1]],top):
                    if re.match("[eEfg]",param["format"][-1]):
                        top = float(top)
                    if re.match("[du]",param["format"][-1]):
                        top = int(top)
                    values=[top]+values
                else:
                    raise ValueError("Error parsing '"+path
                                     +"' into DirectoryStructure variables. '"
                                     +top+"' does not have the correct format ("
                                     +param["format"]+") for '"
                                     +param["variable"]+"'.")
            elif "static" in param:
                if param["static"] == top:
                    values = [top] + values
                else:
                    raise ValueError("Error parsing '"+path
                                     +"' into DirectoryStructure variables. '"
                                     +top+"' does not match static string '"
                                     +param["static"]+"'")
        return values

    def fromPath2Dict(self,trialpath):
        """Returns variables defined by the directory structure as a
        dictionary of key-value pairs.  Using this result with
        fromArgs() will return the original path.
        Args:
            trialpath - path to parse. If a file name is included, it is ignored,
        Returns:
            dictionary of variable names/values.  'static' names are
            not returned.
        Raises:
            ValueError if the path does not conform to the defined
            directory structure.
        """
        #There is no equivalent to scanf in Python so we use the table
        #from the Re manual page
        scanfRe = {"c":	".",
                   "d":	"[-+]?\d+",
                   "e": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
                   "E": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
                   "f": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
                   "g": "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?",
                   "i":	"[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)",
                   "o":	"[-+]?[0-7]+",
                   "s":	"\S+",
                   "u":	"\d+",
                   "x": "[-+]?(0[xX])?[\dA-Fa-f]+", 
                   "X": "[-+]?(0[xX])?[\dA-Fa-f]+"}
        pairs = {}
        if os.path.isfile(trialpath):
            trialpath = os.path.dirname(trialpath)
        path=trialpath
        for param in reversed(self.prefs[self.key]):
            path,top = os.path.split(path)
            if "variable" in param:
                if re.match(scanfRe[param["format"][-1]],top):
                    if re.match("[eEfg]",param["format"][-1]):
                        top = float(top)
                    if re.match("[du]",param["format"][-1]):
                        top = int(top)
                    pairs[param["variable"]]=top
                else:
                    raise ValueError("Error parsing '"+trialpath
                                     +"' into DirectoryStructure variables. '"
                                     +top+"' does not have the correct format ("
                                     +param["format"]+") for '"
                                     +param["variable"]+"'.")
                    
        return pairs

    def _cleanupDirname(self,dirname,param):
        '''
        Apply any search and replace requests to the new directory name
        and replace all directory separators with underscores.
        Args:
            dirname: (str) current name of the directory structure name
            param: (dict) user supplied dictionary with 'search' and 'replace' keys
        Returns:
            updated dirname. If no search and replace is requested,
            the returned value has the same type as dirname
        '''
        if "search" in param and "replace" in param:
            dirname=re.sub(param["search"],param["replace"],str(dirname))
            self.log.debug("Regex search and replace to : "+dirname)
            
        # python 2/3 compatibility to catch unicode strings
        try:
            isinstance(dirname,basestring)
            def isstr(s):
                return isinstance(s,str)
        except NameError as e:
            def isstr(s):
                return isinstance(s,str)

        if isstr(dirname):
            dirname=re.sub(os.sep,r'_',str(dirname))
            self.log.debug("Removing directory separators to create : "+dirname)
        return dirname

    def __iter__(self):
        """
        Iterator.
        """
        #Note: the internal counter has a counter for each parameter
        #that is part of the directory hierarchy.
        self.combinations = self._countCombinations()
        try:
            self.counter = [0]*len(self.combinations)
        except TypeError as e:
            self.log.debug("No combinations. No interations")
            self.counter=None
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        """
        Iterator. Returns the next directory.
        """
        if self.counter is None or ( self.counter[0]>=self.combinations[0]):
            raise StopIteration
        else:
            #construct a list of keyword/value pairs from which to
            #construct the directory string
            kwargs={}
            #regular directories from the user options
            for i,param in enumerate(self.prefs[self.key]):
                if "variable" in param and param["variable"] != "runnumber":
                    kwargs[param["variable"]] = self.args[
                        param["variable"]][self.counter[i]]
                                                
            self._incrCounter()                
            return self.fromArgs(**kwargs)

    def _incrCounter(self):
        """
        Increment the iterator counter.  The right most directory
        iterates the fastest.  The counter then gives the index of
        each user supplied parameter for the order directory
        structure.
        
        When iterating past the last value, self.counter[0] ==
        self.combinations.[0].

        Side Effects:
            Updates the iterator counter
        """
        for i in range(len(self.counter)-1,0,-1):
            if ( self.counter[i]+1 < self.combinations[i]):
                self.counter[i]+=1
                return
            else:
                self.counter[i]=0
        self.counter[0]+=1

    def _countCombinations(self):
        """
        Counts up all of the supplied user parameters. Stores them in
        an array ordered the same as the directory structure.  The
        product of all the array elements gives the total number of
        different paths that can be produced. Static and runnumber
        directories are not included.

        Returns:
           An array containing the number of each dirstruct *variable*.
        """
        combinations = []
        for param in self.prefs[self.key]:
            if "static" in param:
                combinations.append(1)
            if "variable" in param and param["variable"] != "runnumber":
                try:
                    combinations.append(len(self.args[param["variable"]]))
                except TypeError as e:
                    self.log.debug("No arguments for parameter "
                                   +param["variable"]
                                   +" so no combinations can be counted.")
                    return
        return combinations
    
    def _getNextRun(self,path):
        """
        Examines the existing directory structure and returns the next run
        number for this parameter set. Does not check the format of
        existing numbered directories or whether or not they are
        empty.

        Args:
            path - path to examine sans run numbers
        Returns:
            An integer indicating the number of the next run
        """
        self.log.debug("Constructing next run number directory from: "+path)
        path=os.path.join(os.getcwd(),path)
        if os.path.isdir(path):
            subdir = os.listdir(path)
            self.log.debug("Subdirectories in the parent")
            
            numbered_subdir=[]
            for i,file in enumerate(subdir):
                if re.search(r"^\d+",subdir[i]):
                    numbered_subdir.append(file)
            numbered_subdir.sort(key=int)
            if len(numbered_subdir) ==0:
                self.log.debug("No numbered subdirectories found.")
                newdir = 0
            else:
                newdir = int(numbered_subdir[-1])+1
            self.log.debug("Searching for valid restart file")
            while newdir >= 0 and not self._validRestart(path,newdir):
                newdir-=1
            if newdir < 0:
                raise RuntimeError("Could not find a valid restart file for runnumber directory with base : "
                                   +path)
            self.log.info("New numbered directory : "+str(newdir))
            return newdir
        elif os.path.exists(path):
            raise RuntimeError(path+" exists and is not a directory.")
        else:
            if not self._validRestart(path,0):
                raise RuntimeError("Could not find a valid restart file for runnumber directory with base : "
                                   +path)
            return 0

    def _validRestart(self,path,number):
        '''
        Searches for a valid restart file in the previous directory,
        which is either `runnumber-1` or `startdir`, if defined.
        Args:
            path: (str) path that will contain the runnumnber directory
            number: (int) proposed runnumber
        Returns:
            (boolean) True if found, False otherwise
        '''
        self.log.debug('Searching for restart file in "'+path
                       +'" for new number : '+str(number))
        runnumberDict = self.prefs[self.key][-1]
        if "restart" not in runnumberDict:
            self.log.info("Required restart file for runnumber directory not defined. Proceeding.")
            return True
        prevnumber = number-1
        if prevnumber < 0:
            self.log.debug('No previous run directores. Trying startdir')
            try:
                prevpath=os.path.join(
                    path,
                    ("{0:"+runnumberDict["format"]+"}")
                    .format(number),
                    runnumberDict["startdir"])
                self.log.debug('Found startdir')
            except:
                self.log.debug('No startdir. Done.')
                return False
        else:
            prevpath=os.path.join(path,
                                  ("{0:"+runnumberDict["format"]+"}")
                                  .format(prevnumber))
            self.log.debug('Constructed previous run directory: '+
                           prevpath)
        restartfile = os.path.normpath(os.path.join(prevpath,
                                   string.Template(runnumberDict["restart"])
                                   .substitute(self.args)))
        found = os.path.exists(restartfile)
        if found and "success" in runnumberDict:
            self.log.debug('Restart file exists and searching contents')
            found = False
            for line in reverse_readline(restartfile):
                if re.search(runnumberDict['success'], line):
                    found=True
                    self.log.debug('Found success line: '+line)
                    break
        self.log.info(restartfile+" exists : "+str(found))
        return found

    

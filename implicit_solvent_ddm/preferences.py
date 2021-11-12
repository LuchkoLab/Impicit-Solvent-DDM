import os
import collections
import copy
import json
import pickle
import re
import logging
import pprint

#adapted from 
#http://stackoverflow.com/questions/8230315/python-sets-are-not-json-serializable
class PythonObjectEncoder(json.JSONEncoder):
    """Limited support for reading in specially defined 'types' in JSON
    format. In Python it is possible to pass a data type, e.g., str,
    int, float.  JSON does not support this natively so we define a
    special type of object. Objects not of the standard type will have
    their string representation taken and the first two words will
    become a key-value pair (the key will be placed inside "<>"s).

    For example, a str object has the string representation "<type
    'str'>".  This will be written in the JSON as { "<type>" : "str"
    }.  as_python_object() can then reinterpret this.
    """
    def default(self, obj):
        """
        Args:
           obj - whatever Python object is to be written
        Returns:
           Either the default JSONEncoding, if there is one, or it's
           partial string representation
        """
        if isinstance(obj, 
                      (list, dict, str, int, float, bool, type(None))):
            return json.JSONEncoder.default(self, obj)
        m = re.search(r"(\w+) '(\w+)",str(obj))
        return {"<"+m.group(1)+">":m.group(2)}

def as_python_object(dct):
    """Limited support for reading in specially defined 'types' in JSON
    format. In Python it is possible to pass a data type, e.g., str,
    int, float.  JSON does not support this natively so we define a
    special type of object.  If the JSON data is written as
    {"<type>","_some_type_"} then _some_type_ (not the string) is
    returned.  Only str, float and int are supported.
    
    Args:
       dct - a dictionary object
    Returns:
       if dct has the key "<type>" then str, float or int is returned.
       Otherwise dct is returned.
    """
    if '<type>' in dct:
        if dct['<type>'] == 'str':
            return str
        if dct['<type>'] == 'float':
            return float
        if dct['<type>'] == 'int':
            return int
        raise TypeError("'"+dct['<type>']+
                        "' is not a support type for preference files")
    return dct

class Preferences(object):
    """
    Reads in and stores default and user generated preference
    files. Details of the format are found in core.py. The resulting
    preference object can be treated as a dictionary. For example, to
    get the arguments from the preferences, use

    prefs = Preferences()
    print(prefs["arguments"])
    """

    def __init__(self):
        """
        Initialize and read in default preferences.

        Args:
            file - path and file name of default preferences

        Side Effects:
            reads and stores default presences
        """
        #keep a record of files read
        self.files = []
        self.prefs = dict()
        self.log = logging.getLogger(__name__)

    def copy(self):
        '''
        Create a deep copy of the Preference object.
        
        copy.deepcopy() doesn't work.  This takes manually does the deepcopy.

        Returns:
            an independent copy of the original Preference object
        '''
        clone=Preferences()
        clone.files = copy.copy(self.files)
        clone.prefs = copy.deepcopy(self.prefs)
        return clone
    
    def __getitem__(self,key):
        """
        Return a item from the preferences dictionary.
        """
        return self.prefs[key]

    def addFile(self,file):
        """Read another preference file. Overwrite any name clashes.  

        For each file, there are two special top level keys: "submit"
        and "parallel".  For each of these, read in either the file or
        value for each nested key as an additional preference file.

        For example,

        {"submit" : {"local.cmd" : {"template" :"$COMMAND"},
	             "pbs.cmd" : "pbs.json"}
        
        For the key "local.cmd" the template will be associated with
        both ["submit"]["local.cmd"]["template"] and appended to the
        templates.  For the key "pbs.cmd", "pbs.json" will be read in.
        The contents will be associated with both
        ["submit"]["pbs.cmd"] and at the top level.

        A third special top level key is 'include'.  This is only read
        if it is at the top level.  It will read in the include
        file(s) immediately.  If there is a conflict between the
        include file the including file, the including file wins.

        Args:
            file - path and file name of preferences

        Side Effects:
            reads and stores preferences. Overwrites any conflicts

        """
        file = os.path.abspath(file)
        try:
            self.log.debug("Adding file : "+file)
        except:
            self.log.debug("Adding file object")
        prefs = self._readFile(file)

        # recursively process any included preference files
        if 'include' in prefs.keys():
            if isinstance(prefs['include'],str):
                prefs['include'] = os.path.join(os.path.dirname(file),
                                                prefs['include'])
                self.log.debug("Found include : "+prefs['include'])
                self.addFile(prefs['include'])
            elif isinstance(prefs['include'],list):
                self.log.debug("Found includes : "+str(prefs['include']))
                for include in prefs['include']:
                    include = os.path.join(os.path.dirname(file),
                                           include)
                    self.addFile(include)
                
        #first pass. Change paths from relative to prefs file to
        #relative to working directory
        for key,value in self.nested_dict_iter(prefs):
            if "file" in value:
                self.log.debug("Found template file : "+value["file"])
                relative_path = value["file"]
                value["file"]=os.path.join(os.path.dirname(file),
                                           value["file"])
                self.log.debug("Setting file '"+relative_path+"' to : "+value["file"])
                
        #second pass. Copy submit and parallel prefs to the top level
        #so they are read in arguments and templates
        for key,value in prefs.items():#self.nested_dict_iter(prefs):
            if key in ["submit","parallel"]:
                for method in value:
                    #copy the arguments and templates globally
                    #'order' contains special information
                    if method != "order":
                        prefs = self._mergePrefs(prefs,value[method])
        #merge with existing prefs
        self.log.debug("Merging '"+file+"' into preferences")
        self.prefs = self._mergePrefs(self.prefs,prefs)

    def nested_dict_iter(self,nested):
        """
        Generator to recursively walk though the nested dicts
        Args:
           nested - nested dict to walk through
        Returns:
           key/value pairs
        """
        for key, value in nested.items():
            if isinstance(value, collections.Mapping):
                for inner_key, inner_value in self.nested_dict_iter(value):
                    yield inner_key, inner_value
                else:
                    yield key, value

    def _mergePrefs(self,A,B):
        """merges two preference dictionaries.  If there is a conflict then B
        wins.
        Args:
            A - a preference dict
            B - a preference dict
        Returns:
            the union of A and B with B overwriting A in the case of a conflict
        """
        if not isinstance(B,dict):
            return B
        merge = copy.deepcopy(A)
        for key, value in B.items():#.iteritems():
            self.log.debug("merging key : "+key)
            if key in merge and isinstance(merge[key], dict):
                self.log.debug('conflict in key: '+key)
                merge[key] = self._mergePrefs(merge[key],value)
            else:
                self.log.debug("adding key : "+key)
                merge[key] = copy.deepcopy(value)
        return merge

    def _readFile(self,file):
        """Reads in the JSON preference file and returns the result as a dict.
        Args:
            file : (string or file object) name of the file to read
        Returns:
            the contents of the file stored as a dict
        """
        self.files.append(file)
        if isinstance(file,str):
            fh=open(file)
        else:
            fh = file
        try:
            prefs = json.load(fh,object_hook=as_python_object)
        except (ValueError,TypeError) as e:
            try:
                print("ERROR READING "+file)
            except:
                print("ERROR READING FILE OBJECT")                
            raise(e)
        fh.close()
        return dict(list(prefs.items()) + list(prefs.items()))

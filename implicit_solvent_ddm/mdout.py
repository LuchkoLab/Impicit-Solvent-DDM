import pprint as pp
import re
from typing import Type

import pandas as pd

#These are regular expressions for basic types from the Re manual
#page
scanfRe = {"c":	".",
           "d":	"[-+]?\d+",
           "e": "[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
           "E": "[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
           "f": "[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
           "g": "[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
           "i":	"[-+]?(?:0[xX][\dA-Fa-f]+|0[0-7]*|\d+)",
           "o":	"[-+]?[0-7]+",
           "s":	"\S+",
           "u":	"\d+",
           "x": "[-+]?(?:0[xX])?[\dA-Fa-f]+", 
           "X": "[-+]?(?:0[xX])?[\dA-Fa-f]+"}

def to_dataframe(filename)-> pd.DataFrame:
    '''Parse the standard frame-by-frame data from an Amber mdout file
    into a Pandas DataFrame. The DataFrame correctly assigns dtypes
    but does not create an index.

    Args:
        filename:
            name of the mdout file to read
    Returns:
        a Pandas DataFrame
    '''
    # values to pull from each block
    keys = ["NSTEP", "TIME(PS)", "TEMP(K)", "PRESS", "Etot", "EKtot", "EPtot", "BOND",
            "ANGLE", "DIHED", "1-4 NB", "1-4 EEL", "VDWAALS", "EELEC", "EGB",
            "RESTRAINT", "ESURF", "EHBOND", "EKCMT", "VIRIAL", "VOLUME", "Density","ERISM"]
    types = [int] + [str] + [float]*(len(keys)-2)
    keytype = dict(zip(keys,types))
    # dataframe to hold the data
    df = pd.DataFrame(columns=keys)
    # set the types
    for k,t in zip(keys,types):
        df.loc[:,k] = df[k].astype(t)
    # regular expression for any of the keys followed by a number
    key_value_pair_RE = (r"("+"|".join([re.escape(key) for key in keys])
                         +") *= *("+scanfRe['g']+")" )
    rows=[]
    with open(filename,'r') as fh:
        # get each block as a single line
        for block in _dataBlock(fh,"^ NSTEP", "^ --------"):
            # this will be a list of pairs
            pairs = re.findall(key_value_pair_RE,block)
            # covert to a record and make sure we use the correct type
            # or everything becomes an object
            df_items = [(x[0],[keytype[x[0]](x[1])]) for x in pairs]
            # append to the data frame.  It may be faster to do this
            # all at once at the end of the loop
            # df = df.append(pd.DataFrame.from_items(df_items))
            rows.append(pd.DataFrame.from_items(df_items))
    df = df.append(rows)
    df.reset_index(inplace=True)
    return df

def min_to_dataframe(filename)-> pd.DataFrame:
    '''Parse minimization frame-by-frame data from an Amber mdout file
    into a Pandas DataFrame. The DataFrame correctly assigns dtypes
    but does not create an index.

    Args:
        filename:
            name of the mdout file to read
    Returns:
        a Pandas DataFrame
    '''
    # keys for key=value pairs to pull from each block
    keys = ["BOND","ANGLE", "DIHED", "1-4 VDW", "1-4 EEL", "VDWAALS", "EEL", 
            "ERISM","RESTRAINT", "EGB", "ESURF"]
    types = [float]*(len(keys))
    keytype = dict(zip(keys,types))
    # header style keys
    header = ["NSTEP", "ENERGY", "RMS", "GMAX", "NAME", "NUMBER"]
    headertypes = dict(zip(header,[int, float, float, float, str, int]))
    # store rows here before creating dataframe
    rows = []
    # regular expression for any of the keys followed by a number
    key_value_pair_RE = (r"("+"|".join([re.escape(key) for key in keys])
                         +") *= *("+scanfRe['g']+")" )
    with open(filename,'r') as fh:
        # get each block as a single line

        # The data block, in this case, has three different formats.
        # The first part gives the frame number.
        # The second part is table-like with a header and then
        # values. The third part is a list of key=value pairs.

        # this loop gets the two pieces as separate blocks in
        # alternating fashion. These are used to create a single row,
        # which is appended to the larger dataframe.
        for block in _dataBlock(fh,"^(   NSTEP| BOND|minimizing)", "^( *\d+ |minimization|minimizing)",
                                include_end=True):
            if re.search(r"minimizing", block):
                df_items = {"FRAME": int(re.sub("[^0-9]", "", block.split()[-1]))}
            if re.search(r"NSTEP",block):
                # this comes first
                # keys are first half of tokens, data is the second
                tokens = block.split()
                # covert to a record and make sure we use the correct type
                # or everything becomes an object
                # this is a new list
                df_items.update({k:headertypes[k](v) for k,v in zip(tokens[:len(tokens)//2], tokens[len(tokens)//2:])})
            elif re.search(r"BOND", block):
                # this comes second
                # this will be a list of pairs
                pairs = re.findall(key_value_pair_RE,block)
                # covert to a record and make sure we use the correct type
                # or everything becomes an object
                # add to the existing list from above
                df_items.update({k:keytype[k](v) for k,v in pairs})
                # append to the data frame.  It may be faster to do this
                # all at once at the end of the loop
                rows.append(df_items)

    return pd.DataFrame(rows)


def _dataBlock(fh,start,end,
               header_end="^   4.  RESULTS",
               footer_start="^      A V E R A G E S   O V E R",
               include_end=False):
    '''Generator that returns data blocks from the body of the mdout file
    handle. Each data block is returned a single string.

    Args:
        fh:
            filehandle to read from
        start:
            regular expression for the line that starts the block
        end:
            regular expression for the line that ends the block
        header_end:
            regular expression for the line that ends the header. No
            blocks are read before this line
        footer_start:
            regular expression for the line that starts the footer. No
            blocks are read after this line
        include_end: (boolean)
            include 'end' line as part of block
    Returns:
        generator of strings
    '''
    inbody = False
    inblock = False

    # body to the end of simulation data
    for line in fh:
        # get past the header
        if re.search(header_end,line):
            inbody=True
            continue
        if re.search(footer_start,line):
            break

        if inbody:
            if re.search(start,line):
                inblock = True
                block = ""
            if re.search(end,line) and inblock:
                inblock = False
                if include_end:
                    block += line.rstrip()
                yield block
            if inblock:
                block += line.rstrip()

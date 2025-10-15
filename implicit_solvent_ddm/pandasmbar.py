"""Functions to make applying pyMBAR to Pandas DataFrames easier.
"""

"""
The simplest way to use this is 

    import pandasmbar as pdmbar
    
    equil_info = pdmbar.detect_equilibration(df)
    df_subsampled = pdmbar.subsample_correlated_data(df, equil_info)
    results = pdmbar.mbar(df_subsampled)

With this procedure, equilibration frames from the trajectory are
removed and the remaining frames are subsampled to remove time
correlations.

These functions assume that the original DataFrame is arranged in a
specific format.  It is assumed that there are N uniquely named states
that are sampled from and these samples are then evaluated at M
states.  N must a be a subset of M.  I.e., each state you sample at
must also be evaluated at that state and you should evaluate at other
states of interest. Each row is a time step from trajectory.  The
index values indicate the state it was sampled at and the columns give
the state it was evaluated at.  The names are arbitrary, other than
the fact that they must be unique between states and consistent
between the index and column names. Multiindexes and tuples are fine
and may be mixed.

Time step or iteration data should be removed.

An example DataFrame with a MultiIndex index and tuples for column names.

              (1.0, 0.0)  (2.0, 1.0)  (4.0, 2.0)  (8.0, 3.0)  (16.0, 4.0)
    K_k  O_k                                                             
    1.0  0.0    0.886561    0.109950    0.893553   11.134414    56.963445
         0.0    0.255812    0.081066    3.301016   20.879801    86.315138
         0.0    1.194131    6.479063   25.139726   82.642655   246.011715
         0.0    0.000035    1.016838    8.067211   36.201494   128.537129
    ...              ...         ...         ...         ...          ...
    16.0 4.0    6.214973    6.378721    4.654990    1.105076     1.800344
         4.0    8.466118    9.702476    8.945433    4.971827     0.105579
         4.0    8.398398    9.600018    8.806477    4.825838     0.077444
         4.0    7.345085    8.024619    6.718135    2.774063     0.223711


Multiple dataset can be processed as once if columns exist by which to
group the data. A possible case is multiple molecules sampled and
evaluated at the same states. For example, we may have data such as

                      data1       data2       data3
    state group                                    
    data1 a        0.712948    0.619528    0.283933
          a        0.810213    0.871656    0.602021
    ...
    data1 b        0.712010    0.347209    0.212972
          b        0.660189    0.716812    0.834293
    ...                 ...         ...         ...
    data3 a        0.534624    0.338867   -1.259456
          a        0.671498    0.781570   -1.190549
    ...
    data3 b        0.405522    0.032545   -1.190277
          b        0.507182    0.456970   -0.922860

Where there are two sets of data, 'a' and 'b'.


    equil_info = df.groupby('group').apply(lambda x : pdmbar.detect_equilibration(x, nskip=n))
    df_subsampled = pdmbar.subsample_correlated_data(df, equil_info)
    results = pdmbar.mbar(df_subsampled, 'group')

Notice that the first line directly uses Pandas groupby method, the
last line supplies the group columns as an argument, and the middle
line does not directly use groups.

"""
from sys import excepthook

import pandas as pd
import pymbar


def mbar(df, groupby=None, solver_protocol=None):
    """Applies mbar() to grouped or ungrouped data.

    *Ungrouped data*

    Each group is eventually processed individually.  If the data is
    ungrouped, then there is just one group.

    DataFrame Format
    ----------------

    Format of the DataFrame should be as in this example:

              (1.0, 0.0)  (2.0, 1.0)  (4.0, 2.0)  (8.0, 3.0)  (16.0, 4.0)
    K_k  O_k
    1.0  0.0    0.886561    0.109950    0.893553   11.134414    56.963445
         0.0    0.255812    0.081066    3.301016   20.879801    86.315138
         0.0    1.194131    6.479063   25.139726   82.642655   246.011715
         0.0    0.000035    1.016838    8.067211   36.201494   128.537129
    ...              ...         ...         ...         ...          ...
    16.0 4.0    6.214973    6.378721    4.654990    1.105076     1.800344
         4.0    8.466118    9.702476    8.945433    4.971827     0.105579
         4.0    8.398398    9.600018    8.806477    4.825838     0.077444
         4.0    7.345085    8.024619    6.718135    2.774063     0.223711

    where the index contains one or more columns that label the state that
    the configuration was sampled at and the column headings are the state
    label that the reduced potential was evalatuated at.

    How States are Identified
    -------------------------

    Sampled states are identified from index values, which are combined
    into a tuple, matching the format of the column names. I.e.,
    left-right order matters. A single index column may be used instead of
    a multi-index if it has the same format as the column headings.

    Free energies are calculated for all column headings.  The number of
    samples (configurations) from each state is determined by the
    index.

    Missing Data
    ------------

    Samples are not needed from all states. However, samples *must* be
    evaluated at all states for MBAR. I.e., no NaN values.

    *Grouped Data*

    This is useful where there are many systems being analyzed with
    the same states.  E.g., the solvation free energy of many
    molecules is calculated from the vacuum energy and solvated
    energy.

    In addition to the sampling states (index values) and evaluation
    states (column headings), additional columns that specify the
    group may also be present.  These must all be list in the groupby
    parameter.

    Each group has mbar() applied to it.  Free energy and error
    matrices are returned as dataframe with the same layout, except
    for additional index column(s) to specify the group.  A dataframe
    of MBAR objects is also return, with the groups as the index.

    Format of the DataFrame should be as in this example:

                  1.0       2.0       3.0
    K_k O_k
    1.0 0.0  0.210076  0.420153  0.630229
    2.0 0.0  0.393268  0.786536  1.179805
        0.0  0.060052  0.120105  0.180157
    3.0 0.0  0.004408  0.008817  0.013225
        0.0  0.394894  0.789787  1.184681
        0.0  0.167838  0.335676  0.503514
    1.0 1.0  0.154848  0.309697  0.464545
    2.0 1.0  0.059952  0.119904  0.179856
        1.0  0.000276  0.000551  0.000827
    3.0 1.0  0.029513  0.059026  0.088539
        1.0  0.712211  1.424421  2.136632
        1.0  0.004528  0.009056  0.013583

    Here 'O_k' can be used to make two groups.  'K_k' is the sampled
    state and matches the evaluated states in the column headings.

    Args:
        df :
            (DataFrame) dataframe to process
        groupby :
            (sequence) column/index headings to groupby
    Returns:
        (DataFrame, DataFrame, MBAR) DataFrames for the free energies
        differences (Deltaf_ij), error estimates in free energy
        difference (dDeltaf_ij), and the pyMBAR object, which can be
        used to get more detail. Groups are identified using the
        groupby columns.  The index is reset to that of the input
        DataFrame. For the MBAR dataframe, there is no sampled state
        column in the index.  E.g.,

                                                      mbar
        O_k
        0.0    <pymbar.mbar.MBAR object at 0x2ad5d6f97470>
        1.0    <pymbar.mbar.MBAR object at 0x2ad5de36f9e8>

    """

    if groupby is None:
        return _mbar(df)
    else:
        # Remove any groupby columns from the index.  This makes it
        # easier to remove them later.
        # If there is is a standard index and reset_index is passed an
        # empty list, the index is dropped.  I think this is a bug in Pandas.
        groupby_in_index = list(set(df.index.names) & set(groupby))
        if len(groupby_in_index) == 0:
            local_df = df.copy()
        else:
            local_df = df.reset_index(groupby_in_index)
        groups = local_df.groupby(groupby, as_index=False)
        free_energies = []
        errors = []
        mbars = []
        for name, group in groups:
            name = list(name)
            # drop the groupby columns since mbar doesn't want to see them
            group = group.drop(groupby, axis="columns")
            fe, err, mb = _mbar(group)
            # add the groupy info back in
            for frame in [fe, err]:
                frame[groupby] = pd.DataFrame([name], index=frame.index)
            free_energies.append(fe)
            errors.append(err)
            # create a data frame for the MBAR objects with groupy info
            mbars.append(pd.DataFrame([name], columns=groupby))
            mbars[-1]["mbar"] = mb
        # create the dataframe to be returned and reset the index to what we got as input
        free_energies = pd.concat(free_energies).reset_index().set_index(df.index.names)
        errors = pd.concat(errors).reset_index().set_index(df.index.names)
        if len(groupby_in_index) != 0:
            mbars = pd.concat(mbars).set_index(list(set(df.index.names) & set(groupby)))

        return free_energies, errors, mbars


def _mbar(df):
    """Applies MBAR (pyMBAR) to the DataFrame. Does not handle grouping.

    See `mbar()`.
    Args:
        df :
            (DataFrame) dataframe to process
    Returns:
        (DataFrame, DataFrame, MBAR) DataFrames for the free energies
        differences (Deltaf_ij), error estimates in free energy
        difference (dDeltaf_ij), and the pyMBAR object, which can be
        used to get more detail.

        Free energy and error data frames attempt to preserve the
        input index and column names.  So, the free energy data frame
        should look something like

                  (1.0, 0.0)  (2.0, 1.0)  (4.0, 2.0)  (8.0, 3.0)  (16.0, 4.0)
        K_k  O_k
        1.0  0.0    0.000000    0.310918    0.662005    0.936818     1.212012
        2.0  1.0   -0.310918    0.000000    0.351087    0.625900     0.901094
        4.0  2.0   -0.662005   -0.351087    0.000000    0.274813     0.550007
        8.0  3.0   -0.936818   -0.625900   -0.274813    0.000000     0.275194
        16.0 4.0   -1.212012   -0.901094   -0.550007   -0.275194     0.000000
    """
    # count the number samples from each state
    N_k = df.groupby(df.index.names).count().iloc[:, [0]]

    # check for states with no samples since these are not
    # counted.

    # flatten the index so it matches the column names
    N_k.index = N_k.index.tolist()

    # merge this with a dataframe that has the states from the columns
    # as the index. 'outer' means that any states without samples will
    # now have a count of NaN.
    N_k = pd.merge(
        pd.DataFrame(index=df.columns),
        N_k,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # replace the NaNs with 0 and match the order of the column states
    N_k = N_k.fillna(0).reindex(df.columns)

    # solver_protocol= (dict(method="L-BFGS-B"), )
    try:
        #try to use 'hybr' solver 
        mbar = pymbar.MBAR(df.values.T, N_k.values.flatten())
        results = mbar.compute_free_energy_differences()
    except:
        # fails to solve switch to `robust` solver 
        mbar = pymbar.MBAR(
            df.values.T,
            N_k.values.flatten(),
            solver_protocol="robust",
        )
        results = mbar.compute_free_energy_differences()

    free_energies = pd.DataFrame(results["Delta_f"], columns=df.columns.values)
    try:
        free_energies.index = pd.MultiIndex.from_tuples(
            df.columns, names=df.index.names
        )
    except:
        free_energies.index = pd.Index(df.columns, name=df.index.name)
    # free_energies.index = df.columns
    # free_energies.index.names = ['states']

    errors = pd.DataFrame(results["dDelta_f"], columns=df.columns)
    try:
        errors.index = pd.MultiIndex.from_tuples(df.columns, names=df.index.names)
    except:
        errors.index = pd.Index(df.columns, name=df.index.name)
    # errors.index = df.columns
    # errors.index.names = ['states']

    return free_energies, errors, mbar


def detect_equilibration(df, nskip=1):
    """Apply pyMBAR equilibration detection of time series to a DataFrame.

    Data is expected to have the same format as is used for pyMBAR.

    DataFrame Format
    ----------------

    Format of the DataFrame should be as in this example:

                 (data1, a)  (data2, b)  (data3, a)
    state group
    data1 a        0.712948    0.619528    0.283933
          a        0.810213    0.871656    0.602021
          a        0.712010    0.347209    0.212972
          a        0.660189    0.716812    0.834293
    ...                 ...         ...         ...
    data3 a        0.534624    0.338867   -1.259456
          a        0.671498    0.781570   -1.190549
          a        0.405522    0.032545   -1.190277
          a        0.507182    0.456970   -0.922860

    where the index contains one or more columns that label the state that
    the configuration was sampled at and the column headings are the state
    label that the reduced potential was evalatuated at.

    As there are multiple timeseries with multiple energies,
    equilibration detection is applied to the sampled state using the
    energy for that state.  E.g., in the example above, there are five
    time series (indicated by the unique index values) and each has a
    data column associated with it.  The output in this case may look like

                     t          g    Neff_max
    state group
    data1 a        0.0  23.096191   43.340481
    data2 b        0.0  18.703865  106.983238
    data3 a      330.0   9.135793   18.717588


    Groupby can be applied as in

    equil = df.groupby(columns).apply(lambda x : detect_equilibration(x, nskip=n))

    Args:
        nskip :
            (int) use this stride between samples to reduce the number
            of samples and total computation time.

    Returns:
        (dataframe) each state+group has row, giving the (t) initial
        index that equilibrate data starts from, (g) statistical
        inefficiency of the data, and (Neff_max) number of
        uncorrelated samples.
    """

    groupby = df.groupby(df.index)

    # for each index group, run equilibration detection on the column that matches the name
    equil_info = {}
    for name, group in groupby:
        equil_info[name] = pymbar.timeseries.detect_equilibration(
            group[name].values, nskip=nskip
        )

    # build the output dataframe
    equil_info = pd.DataFrame(equil_info, index=["t", "g", "Neff_max"]).T
    equil_info.t = equil_info.t.astype("int")
    equil_info.index.names = df.index.names
    return equil_info


def subsample_correlated_data(df, equil_info=None, conservative=False):
    """Apply pyMBAR subsampling of time series to a DataFrame.

    Data is expected to have the same format as is used for pyMBAR.

    DataFrame Format
    ----------------

    Format of the DataFrame should be as in this example:

                 (data1, a)  (data2, b)  (data3, a)
    state group
    data1 a        0.712948    0.619528    0.283933
          a        0.810213    0.871656    0.602021
          a        0.712010    0.347209    0.212972
          a        0.660189    0.716812    0.834293
    ...                 ...         ...         ...
    data3 a        0.534624    0.338867   -1.259456
          a        0.671498    0.781570   -1.190549
          a        0.405522    0.032545   -1.190277
          a        0.507182    0.456970   -0.922860

    where the index contains one or more columns that label the state that
    the configuration was sampled at and the column headings are the state
    label that the reduced potential was evalatuated at.

    As there are multiple timeseries with multiple energies,
    equilibration detection is applied to the sampled state using the
    energy for that state.  E.g., in the example above, there are five
    time series (indicated by the unique index values) and each has a
    data column associated with it.  The output will have the same
    format as the input dataframe, except that the rows have been
    removed, consistent with the `equil_info` parameter.

    Args:
        equil_info :
            (dataframe) The index should have same columns as the
            input dataframe, with one row for each sampled state.  It
            should have 't' (starting index) and 'g' (sampling
            inefficiency) columns.  `detect_equilibration()` generates
            a dataframe of the correct format.
        conservative :
            (bool) sample uniformly with an interval of `ceil(g)` if
            `True`. Otherwise, sample with an intervale of
            approximately `g`.

    Returns:
        (dataframe) each state+group has row, giving the (t) initial
        index that equilibrate data starts from, (g) statistical
        inefficiency of the data, and (Neff_max) number of
        uncorrelated samples.

    """
    groupby = df.groupby(df.index)

    if equil_info is None:
        equil_info = pd.DataFrame(
            {"t": [0] * len(groupby), "g": [None] * len(groupby)}, index=groupby.groups
        )

    # for each index group, subsample the trajectory (rows)
    subsampled = []
    for name, group in groupby:
        indices = pymbar.timeseries.subsample_correlated_data(
            group[name].values[
                equil_info.t.loc(axis=0)[name] :
            ],  # get the values for the column that matches 'name', select points starting from t
            g=equil_info.g.loc(axis=0)[name],  # use the supplied sampling inefficiency
            conservative=conservative,
        )
        subsampled.append(group.iloc[indices])
    # build the output dataframe
    return pd.concat(subsampled, sort=False)

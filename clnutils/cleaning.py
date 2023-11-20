import numpy as np
from pandas import DataFrame, concat, to_numeric
from clnutils import super_sub_scriptreplace
from math import isnan
import re


# NEED TO SPLIT INTO TWO SEPERATE FILES,
# ONE FOR PRE-UPLOAD CLEANING (I.E. ON FULL DATASET)
# ONE FOR POST-UPLOAD CLEANING (I.E. ON SUBSET OF DATASET AFTER
# SELECTING A SPECIFIC SAMPLE TYPE)


def remove_missing_depth(
    df,
    dfrom="from",
    dto="to",
    notes_path=None,
    df_name="ENV DB",
):
    """Removes rows with missing depth values
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data

    dfrom : str, optional, default 'from'
        column name of start depth

    dto : str, optional, default 'to'
        column name of end depth

    notes_path : str, optional, default None
        path to notes file to write to, if None does not write to file
    Returns
    -------
    pandas DataFrame
        DataFrame with missing depth rows removed
    """
    # convert to numeric
    for col in [dfrom, dto]:
        df[col] = to_numeric(df[col], errors="coerce")

    n_missing_from_to = df.loc[df[dfrom].isna() & df[dto].isna()].shape[0]
    n_missing_from_not_to = df.loc[df[dfrom].isna() & ~df[dto].isna()].shape[0]
    n_missing_to_not_from = df.loc[~df[dfrom].isna() & df[dto].isna()].shape[0]
    # drop rows with missing depth
    df = df.loc[df[dfrom].notna()].copy().reset_index(drop=True)
    df = df.loc[df[dto].notna()].copy().reset_index(drop=True)
    # write to notes file
    if notes_path is not None:
        with open(notes_path, "a+") as note:
            note.write(
                f"{df_name} Number of samples with missing start depth: {n_missing_from_not_to}\n"
            )
            note.write(
                f"{df_name} Number of samples with missing end depth: {n_missing_to_not_from}\n"
            )
            note.write(
                f"{df_name} Number of samples with missing start and end depth: {n_missing_from_to}\n"
            )
            note.write(
                f"{df_name} Number of samples remaining after missing depth droped: {len(df)}\n"
            )

    return df


def remove_duplicates(
    df,
    dupe_col="unique_duplicate",
    metal_start="al_pct",
    metal_end="zr_ppm",
    aba_start="aba",
    aba_end="metals",
    notes_path=None,
    df_name="ENV DB",
):
    """Removes duplicate rows from a dataframe
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data

    dupe_col : str, optional, default 'unique_duplicate'
        column name of unique duplicate column

    metal_start : str, optional, default 'al_pct'
        column name of first metal column

    metal_end : str, optional, default 'zr_ppm'
        column name of last metal column

    aba_start : str, optional, default 'aba'
        column name of first ABA column

    aba_end : str, optional, default 'metals'
        column name of last ABA column

    notes_path : str, optional, default None
        path to notes file to write to, if None does not write to file

    Returns
    -------
    pandas DataFrame
        DataFrame with duplicate rows removed
    """
    # drop duplicates (i.e. same data in the row)
    sh0 = df.shape[0]  # original number of rows
    df = (
        df.loc[df[dupe_col] != "Duplicate"].copy().reset_index(drop=True)
    )  # drop duplicates from dupe_col
    sh1 = df.shape[0]  # number of rows after dropping duplicates from dupe_col
    df.drop_duplicates(inplace=True)  # drop all duplicates
    df.dropna(how="all", axis=0)  # drop rows where all values are NaN
    df.dropna(how="all", axis=1)  # drop columns where all values are NaN
    df.reset_index(inplace=True, drop=True)  # reset index
    sh2 = df.shape[0]  # number of rows after dropping all duplicates

    # drop duplicated metals and ABA
    df = df.drop_duplicates(
        subset=df.loc[0, metal_start:metal_end].index
    )  # drop duplicated metals
    sh3 = df.shape[0]  # number of rows after dropping duplicated metals
    df.reset_index(inplace=True, drop=True)  # reset index
    df = df.drop_duplicates(
        subset=df.loc[0, aba_start:aba_end].index
    )  # drop duplicated ABA
    sh4 = df.shape[0]  # number of rows after dropping duplicated ABA
    df.reset_index(inplace=True, drop=True)  # reset index
    # drop rows where all metals NaN
    df = df.dropna(
        how="all", subset=df.loc[0, metal_start:metal_end].index
    ).reset_index(
        drop=True
    )  # drop rows where all metals NaN
    sh5 = df.shape[0]  # number of rows after dropping all NaN metals

    # write to notes file
    if notes_path is not None:
        with open(notes_path, "a+") as note:
            note.write(
                f"{df_name} Number of samples with named duplicates: {sh0 - sh1}\n"
            )
            note.write(
                f"{df_name} Number of samples remaining after named duplicates dropped: {sh1}\n"
            )
            note.write(f"{df_name} Number of duplicated rows ENV DB: {sh1-sh2}\n")
            note.write(
                f"{df_name} Number of samples remaining after duplicate rows droped: {sh2}\n"
            )
            note.write(f"{df_name} Samples with duplicated metals ENV DB: {sh2-sh3}\n")
            note.write(f"{df_name} Samples with duplicated ABA ENV DB: {sh3-sh4}\n")
            note.write(
                f"{df_name} Number of samples remaining after metal and ABA duplicates dropped: {sh4}\n"
            )
            note.write(f"{df_name} Number samples with all NaN in metals: {sh4-sh5}\n")
            note.write(
                f"{df_name} Number samples remaining after all NaN metals droped: {sh5}\n"
            )

    return df


def drop_target_col_na(
    df,
    target_col_individual=None,
    target_col_group=None,
    notes_path=None,
    df_name="ENV DB",
):
    """Drop rows where a given target column is nan or all columns from a target group are nan. Can take multiple individual columns and or multiple groups

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data

    target_col_individual : list-like
        column name or list of column names to drop rows where the column is nan

    target_col_group : list-like
        list of lists of column names to drop rows where all columns in the list are nan

    Returns
    -------
    pandas DataFrame
        DataFrame with rows dropped
    """
    shp0 = df.shape[0]  # original number of rows
    # drop rows where target_col_individual is nan
    if isinstance(target_col_individual, str):
        target_col_individual = [target_col_individual]

    if isinstance(target_col_group, str):
        target_col_group = [target_col_group]

    if target_col_individual is None:
        target_col_individual = []
    for col in target_col_individual:
        df = df.dropna(subset=[col], how="all").reset_index(drop=True)

    if target_col_group is None:
        target_col_group = []
    # drop rows where all columns in target_col_group are nan
    for col_group in target_col_group:
        if isinstance(col_group, str):
            print("Warning single column passed to 'target_col_group', skipping...")
            continue
        df = df.dropna(subset=col_group, how="all").reset_index(drop=True)

    shp1 = df.shape[0]  # number of rows after dropping rows
    # write to notes file
    if notes_path is not None:
        with open(notes_path, "a+") as note:
            note.write(
                f"{df_name} Number of rows with missing target column(s): {shp0-shp1}\n"
            )
            note.write(
                f"{df_name} Number of samples remaining after dropping rows with missing target column(s): {shp1}\n"
            )
    return df


def overlap(
    df,
    hole_col="drill_hole",
    dfrom="from",
    dto="to",
    samp_id1="sample_id",
    intv="interval",
):
    """Assesses drillhole overlap between samples.
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data
    hole_col : str, optional, default 'drill_hole'
        column name of hole_id
    dfrom : str, optional, default 'from'
        column name of start depth
    dto : str, optional, default 'to'
        column name of end depth
    samp_id1 : str, optional, default 'sample_id'
        column name of sample id
    intv : str, optional, default 'interval'
        column name of interval distance, to have overlap calculated
        from the data pass empty string ''
    Returns
    -------
    pandas DataFrame
        DataFrame of overlap data with columns:
        'ovlp_up': str, sample_id of upper sample
        'ovlp_lwr': str, sample_id of lower sample
        'ovlp_dist': numeric, distance of overlap
        'pct_ovlp_up': ovlp_dist over interval distance of upper sample
        'pct_ovlp_lwr': ovlp_dist over interval distance of lower sample
    """
    holes = df[hole_col].unique()  # get the unique holes in the df

    cols = [hole_col, dfrom, dto, samp_id1]

    if intv not in df.columns.tolist():
        intpresent = False
    else:
        cols.append(intv)
        intpresent = True

    temp_d = {
        "ovlp_up": [],
        "ovlp_lwr": [],
        "ovlp_dist": [],
        "pct_ovlp_up": [],
        "pct_ovlp_lw": [],
    }

    for dh in holes:  # for each hole determine if there is overlap
        # temp df of each hole
        temp = df.loc[df[hole_col] == dh, cols]
        # sort by start depth
        temp = temp.sort_values(by=dfrom)
        # reset index
        temp.reset_index(inplace=True)

        # traverse sub-df row by row
        for idx in range(len(temp) - 1):
            # if current row "dto" greater than next row "dfrom" flag it
            if temp.loc[idx, dto] > temp.loc[idx + 1, dfrom]:
                # get interval distance of the two rows
                if intpresent:
                    intv_0 = temp.loc[idx, intv]
                    intv_1 = temp.loc[idx + 1, intv]
                else:
                    intv_0 = temp.loc[idx, dto] - temp.loc[idx, dfrom]
                    intv_1 = temp.loc[idx + 1, dto] - temp.loc[idx + 1, dfrom]

                # calculate overlap distance
                temp_diff = temp.loc[idx, dto] - temp.loc[idx + 1, dfrom]

                # save to dictionary
                temp_d["ovlp_up"].append(temp.loc[idx, samp_id1])
                temp_d["ovlp_lwr"].append(temp.loc[idx + 1, samp_id1])
                temp_d["ovlp_dist"].append(temp_diff)
                temp_d["pct_ovlp_up"].append(temp_diff / intv_0)
                temp_d["pct_ovlp_lw"].append(temp_diff / intv_1)
    overlap_idx = DataFrame(temp_d)
    return overlap_idx


def id_ovlp_to_drop(
    df,
    pct_up="pct_ovlp_up",
    pct_lwr="pct_ovlp_lw",
    samp_up="ovlp_up",
    samp_lwr="ovlp_lwr",
    ovlp_over_cutoff=None,
):
    """Identifies which of overlaping samples to drop
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing upper and lower sample names,
        percent of overlap for both upper and lower samples
    pct_up : str, optional, default 'pct_ovlp_up'
        column name of overlap percent of upper sample
    pct_lwr : str, optional, default 'pct_ovlp_lw'
        column name of overlap percent of lower sample
    samp_up : str, optional, default 'ovlp_up'
        column name of upper sample name
    samp_lwr : str, optional, default 'ovlp_lwr'
        column name of lower sample name
    ovlp_over_cutoff : list_like, optional, default None
        optional, boolean list to mask input df by, if None
        df will be used, if list passed df will be masked
        by ovlp_over_cutoff, keeping only True indexes
    Returns
    -------
    pandas DataFrame
        DataFrame of overlap data with columns
    """
    if ovlp_over_cutoff is not None:
        temp = df[ovlp_over_cutoff].copy().reset_index(drop=True)
    else:
        temp = df.copy().reset_index(drop=True)

    temp["sample_to_drop"] = np.where(
        temp[pct_up] > temp[pct_lwr], temp[samp_up], temp[samp_lwr]
    )

    return temp


def get_discontinuity(exp_df, holeid="drill_hole", dfrom="from", dto="to"):
    """Finds any discontinuities in downhole drill record
    Parameters:
    -----------
    exp_df - pandas dataframe
        contains at minimum holeid, sample start, and sample end
    holeid - str, default 'drill_hole'
        column name of hole id column
    dfrom - str, default 'from'
        column name of sample start column, column needs to be numeric
    dto - str, default 'to'
        column name of sample end column, column needs to be numeric
    Returns:
    --------
    discontinuity - pandas dataframe
        with observations where discontinuities are present. Columns = [holeid,dto,dfrom],
        'to' column is start of discontinuity, 'from' column is end of discontinuity
    """

    # instantiate empty dataframe
    discontinuity = DataFrame()

    # loop through for each holeid in df
    for hole in exp_df[holeid].unique():
        # make temp copy
        temp = exp_df[exp_df[holeid] == hole].copy().reset_index(drop=True)
        # sort df by start column, order matters
        temp.sort_values(by=dfrom, inplace=True)
        # shift dfrom column by -1 to line up with dto column
        temp[dfrom] = temp[dfrom].shift(-1)
        # drop resultant NaN row
        temp.dropna(inplace=True)
        # if realigned dfrom =/= dto mark it
        temp["match"] = temp[dfrom] == temp[dto]
        # record only non-matches
        discontinuity = concat([discontinuity, temp[temp.match == 0]])
    # reindex columns for readability
    return discontinuity.reindex(columns=[holeid, dto, dfrom])


def get_bounds(df):
    """Gets the overall min and overall max of a df
    Parameters
    ----------
    df : pandas DataFrame
        all numeric columns
    Returns
    -------
    tuple
        tuple of min and max values, length 2
    """
    return df.min().min(), df.max().max()


def no_overlapping(x1, x2, y1, y2):
    """Tests for overlap between two sets of numerical values
    Parameters
    ----------
    x1 : numerical
        minimum of group 1
    x2 : numerical
        maximum of group 1
    y1 : numerical
        minimum of group 2
    y2 : numerical
        maximum of group 2
    Returns
    -------
    bool
        True if groups do not overlap, else false
    """
    return max(x1, y1) >= min(x2, y2)  # type: ignore


def find_overlap(minval, maxval, targetfrom, targetto):
    """From two series finds where there is overlap with a min and max value
    Parameters
    ----------
    minval : numerical
        minimum value
    maxval : numerical
        maximum value
    targetfrom : list-like
        numerical values for lower bound
    targetto : list-like
        numerical values for upper bound
    Returns
    -------
    tuple
        two list-like objects with boolean values
    """
    return np.where(targetfrom < maxval, True, False), np.where(
        targetto > minval, True, False
    )


def test_continuity(
    discontinuity,
    env_holes,
    expholeid="drill_hole",
    expfrm="from",
    expto="to",
    envholeid="drill_hole",
    envfrm="from",
    envto="to",
):
    """Given two datasets with sample distance data, one with known discontinuities,
        identifies if and where the second dataset has samples from within those
        discontinuities
    Parameters
    ----------
    discontinuity : pandas DataFrame
        dataframe with known discontinuities, has at minimum hole_id, sample_start,
        sample_end
    env_holes : pandas DataFrame
        test dataframe, discontinuities unknown, has at minimum hole_id, sample_start,
        sample_end
    expholeid : str, optional
        column name of hole_id column for discontinuity df, by default 'drill_hole'
    expfrm : str, optional
        column name of sample_start column for discontinuity df, by default 'from'
    expto : str, optional
        column name of sample_enc column for discontinuity df, by default 'to'
    envholeid : str, optional
        column name of hole_id column for env_holes df, by default 'drill_hole'
    envfrm : str, optional
        column name of sample_start column for env_holes df, by default 'from'
    envto : str, optional
        column name of sample_enc column for env_holes df, by default 'to'
    Returns
    -------
    no_data_env : list
        list of rows from env_holes df that have discontinuities,
        includes columns of sampleid, from, and to
    no_data_exp : list
        list of rows from discontinuity df that have discontinuities,
        includes columns of sampleid, from, and to
    as a tuple with (no_data_env,no_data_exp)
    """
    # instantiate two empty lists
    no_data_env = []
    no_data_exp = []

    # loop through each hole_id in the discontinuity df
    for hole in discontinuity[expholeid].unique():
        # print working hole for tracking
        print(f"Working on {hole}")
        # make copy of subset of each df
        tempexp = discontinuity[discontinuity[expholeid] == hole].copy()
        tempenv = env_holes[env_holes[envholeid] == hole].copy()
        # get maximum and minimum numerical values from the two dfs
        exp_min, exp_max = get_bounds(tempexp[[expfrm, expto]])
        env_min, env_max = get_bounds(tempenv[[envfrm, envto]])
        # determine if any overlap b/t the dfs present
        if no_overlapping(exp_min, exp_max, env_min, env_max):
            # if no overlap skip this iteration
            print("Distances do not overlap\n")
            continue
        # isolate observations in test df where potential overlap could exist
        # (reduces processing time)
        tempenv["under"], tempenv["over"] = find_overlap(
            exp_min, exp_max, tempenv[envfrm], tempenv[envto]
        )
        tempenv["either"] = np.where(
            (tempenv.under == 1) | (tempenv.over == 1), True, False
        )
        # make subset of observations where potential overlap could exist
        tempenv = tempenv[tempenv["either"] == True].copy().reset_index(drop=True)
        # loop through each observation in test df
        for idx, row in tempenv.iterrows():
            # for each observaiton in test df loop through known discontinuities
            for idx_x, row_x in tempexp.iterrows():
                # test if overlap exists on observation by observation level
                if not no_overlapping(
                    row[envfrm], row[envto], row_x[expto], row_x[expfrm]
                ):
                    # if overlap append to list
                    no_data_env.append(row)
                    no_data_exp.append(row_x)
    # return any overlaps that exist
    return no_data_env, no_data_exp


def rename_cols(torename, repldict=super_sub_scriptreplace):
    """Renames object to be compatible with RDBMS
    Parameters
    ----------
    torename : pandas DataFrame
        dataframe to be renamed
    repldict : dict
        dictionary of characters to replace in column names,
        default replaces superscripts and subscripts with
        their respective characters, e.g. H₂O becomes H2O
    Returns
    -------
    pandas dataframe
        dataframe with renamed columns, if a dataframe
        was passed modifies in place, else returns a copy
        of original data as a dataframe
    """
    # this function replaces select characters in column names
    # with more pythonic characters inclusing only alphanumeric and _
    # characters and makes all characters lowercase

    if not isinstance(torename, DataFrame):
        try:
            torename = DataFrame(torename)
        except Exception as e:
            print(f"Could not convert {torename} to DataFrame")
            raise e

    torename = torename.replace("%", "pct", regex=True)
    torename = torename.replace("(?i)gpt", "ppm", regex=True)
    torename = torename.replace(r"\(|\)", "", regex=True)
    torename = torename.replace(r"[^\w\d_]", "_", regex=True)
    if isinstance(repldict, dict):
        torename = torename.replace(super_sub_scriptreplace, regex=True)
    torename = torename.replace(r"_{2,}", "_", regex=True)
    torename = torename.replace(r"^\s*$", np.nan, regex=True)
    for col in torename.columns:
        torename[col] = torename[col].str.lower()

    if torename.shape[1] == 1:
        return torename.squeeze()
    else:
        return torename


def combine_names(nmcol1, nmcol2):
    """Combines two columns of names into a single column
        ignores NaN values, if one column is a substring of the other,
        the longer column is used, if two names are joined they are seperated
        by a double underscore

    Parameters
    ----------
    nmcol1 : list-like
        first column of names
    nmcol2 : list-like
        second column of names
    Returns
    -------
    list
        combined column of names
    """
    combocols = []
    for a, b in zip(nmcol1, nmcol2):
        if isinstance(b, float) and isnan(b):
            combocols.append(a)
        elif isinstance(a, float) and isnan(a):
            combocols.append(b)
        elif str(a) in str(b):
            combocols.append(b)
        elif str(b) in str(a):
            combocols.append(a)
        else:
            # merge the two columns with double underscore
            combocols.append(str(a) + "__" + str(b))
    return combocols


def make_numeric(df, subset=None, as_neg=True, additional=None, exclude=None):
    """
    Identifies numeric strings preceeded by < or > and turns them into numeric
    type. If converstion fails will return NaN for that observation.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        Subset of columns to apply numeric to. If none filters for
        columns with ppm, gpt, pct, ppb, or ppt in the name and does
        a regex search for the form 'kg*t'.
        To override pass an empty list.

    as_neg : bool, default True
        Whether to convert observations with < or > to negative
        values.

    additional : list-like, default None
        Additional substrings to search for in column names to apply
        numeric conversion to.

    exclude : list-like, default None
        List of substrings to exclude from numeric conversion.

    Returns
    -----
    None
        Changes the data in place.
    """
    if additional is None:
        additional = []
    if exclude is None:
        exclude = []
    substrsearch = ["ppm", "gpt", "pct", "ppb", "ppt"] + additional
    if subset is None:
        subset = [col for col in df.columns if any(sub in col for sub in substrsearch)]
        subset = subset + find_kgt(df.columns.drop(subset).tolist())
    elif len(subset) == 0:
        subset = df.columns
    if len(exclude) > 0:
        subset = [col for col in subset if not any(sub in col for sub in exclude)]
    for col in df[subset].select_dtypes("O"):
        if as_neg:
            df[col] = to_numeric(
                # replace spaces and < or > with - to make negative
                # and coerce to numeric
                # converty to string to get around mixed types
                df[col]
                .astype(str)
                .str.replace(" ", "")
                .str.replace(r"<|>", "-", regex=True),
                errors="coerce",
            )
        else:
            df[col] = to_numeric(
                # replace spaces and < or > with "" to make positive
                # and coerce to numeric
                # converty to string to get around mixed types
                df[col]
                .astype(str)
                .str.replace(" ", "")
                .str.replace(r"<|>", "", regex=True),
                errors="coerce",
            )


def test_for_neg(df, subset=None, additional=None, exclude=None):
    """Tests for negative values in a dataframe
        if negative values are found prompts user to continue
        or raise an exception. To be used in tandem with 'make_numeric'
    Parameters
    ----------
    df : pandas dataframe
    subset : list-like, default None
        Subset of columns to apply numeric to. If none filters for
        columns with ppm, gpt, or pct in the name. To override pass
        an empty list.
    additional : list-like, default None
        Additional substrings to search for in column names to apply
        negative testing to.
    exclude : list-like, default None
        List of substrings to exclude from negative testing.
    Returns
    -----
    None
    """
    if additional is None:
        additional = []
    if exclude is None:
        exclude = []
    negcols = []
    substrsearch = ["ppm", "gpt", "pct", "ppb", "ppt"] + additional
    if subset is None:
        subset = [col for col in df.columns if any(sub in col for sub in substrsearch)]
        subset = subset + find_kgt(df.columns.drop(subset).tolist())
    elif len(subset) == 0:
        subset = df.columns
    if len(exclude) > 0:
        subset = [col for col in subset if not any(sub in col for sub in exclude)]
    for col in df[subset].select_dtypes("O"):
        if any(df[col].astype(str).str.startswith("-")):
            negcols.append(col)
        if any(df[col].astype(str).str.contains(r"^<|^>")):
            negcols.append(col)
    for col in df[subset].select_dtypes("number"):
        if any(df[col] < 0):
            negcols.append(col)

    try:
        assert len(negcols) == 0
    except:
        inpt = input("Negative values found. Do you want to continue? [y/n]")
        if "y" in inpt.lower():
            print("Continuing")
            print("Negative values found in columns: ", negcols)
            pass
        else:
            print("Negative values found in columns: ", negcols)
            raise Exception("Negative values found user chose not to continue")


def find_kgt(colist):
    matches = []
    for col in colist:
        if len(re.findall(r"kg[a-zA-Z0-9_]*t", col)) > 0:
            matches.append(col)

    return matches

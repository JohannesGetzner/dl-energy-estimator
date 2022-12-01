import pandas as pd
import re
from warnings import warn


def parse_codecarbon_output(path, save_to_csv=True) -> pd.DataFrame:
    """
    takes the raw codecarbon output, parses the project_name column and returns a pandas DataFrame
    :param path: the path to the file that should be to be parsed
    :param save_to_csv: if True the method will save the parsed data as a csv to the same path with a '-parsed' suffix
    :return: a pandas.DataFrame containing the parsed data
    """
    df = pd.read_csv(path)
    # count the number of properties saved in the 'project_name' column
    col_names = [s.split(':', 1)[0] for s in df.project_name.iloc[0].split(',')]
    df[col_names] = df["project_name"].str.split(pat=",", n=len(col_names) - 1, expand=True)
    for col in col_names:
        df[col] = df[col].str[df[col].iloc[0].find(":") + 1:]
        if df[col].iloc[0].isnumeric():
            df[col] = pd.to_numeric(df[col])

    # parse freeText column
    # find all positional arguments in freeText column
    new_columns = {}
    for idx, row in df.iterrows():
        arg_names = [free_text_arg.split('=')[0] for free_text_arg in row.freeText.split(';')]
        for arg_name in arg_names:
            if arg_name != '':
                new_columns[arg_name] = []
    # fill new columns from freeText column
    for idx, row in df.iterrows():
        if row.freeText != '':
            # split and extract argument names and values from row
            free_text_args = {free_text_arg.split('=')[0]: free_text_arg.split('=')[1] for free_text_arg in
                              row.freeText.split(';')}
            for new_col in new_columns.keys():
                if new_col in free_text_args.keys():
                    new_columns[new_col].append(free_text_args[new_col])
                else:
                    new_columns[new_col].append(None)
        else:
            # no entry in row -> fill all columns with an extra None
            for new_col in new_columns.keys():
                new_columns[new_col].append(None)

    for new_col_name, new_col in new_columns.items():
        df[new_col_name] = new_col
    df.drop(['project_name', 'freeText'], axis=1, inplace=True)
    if save_to_csv:
        df.to_csv(re.sub(r'(.+)(-raw)(.csv)', r'\1' + '-parsed' + r'\3', path))
    return df


def preprocess_and_normalize_energy_data(df, param_cols, aggregate=True, verbose=False) -> pd.DataFrame:
    """
    this function normalizes the measured energy_values by the number of forward-passes and aggregates repeated configs
    :param verbose: set to true for additional information printed to console
    :param df: the pd.DataFrame containing the data from the parsed codecarbon output
    :param param_cols: the parameter names of the module configuration
    :param aggregate: whether to compute the mean-energy of configurations that are identical
    :return: the preprocessed pd.DataFrame
    """
    # TODO: implemented check by slurm-output parsing here
    if (df["cpu_energy"] < 0).any():
        previous_shape = df.shape
        df = df.loc[(df["cpu_energy"] > 0) & (df['gpu_energy'] > 0)]
        warn(f"{previous_shape[0] - df.shape[0]} negative energy values detected! These data points have been removed.")
    energy_cols = [col_name for col_name in df.columns if 'energy' in col_name]
    for col in energy_cols:
        df[col] = df[col].div(df['forward_passes'])
    if aggregate:
        previous_shape = df.shape
        df = df.groupby(param_cols, sort=False).mean(numeric_only=True)
        df.reset_index(inplace=True)
        if verbose:
            print(
                f"Shape before aggregation: {previous_shape}, after aggregation: {df.shape} (non numeric columns removed)")
    return df


def parse_slurm_output_for_errors(path='') -> []:
    # TODO: this function needs to be adjusted to fit the new slurm logs
    """
    checks the slum cluster log for codecarbon errors and returns configurations that which caused the errors
    :param path: the path to the slurm output file
    :return:the invalid configurations
    """
    file = open(path, 'r')
    lines = file.readlines()
    invalid_exps = []
    current_exp = None
    error_checked = False
    for line in lines:
        if line.startswith(" ----"):
            current_exp = line
            error_checked = False
        elif 'ERROR' in line and not error_checked:
            invalid_exps.append(current_exp)
            error_checked = True
    parsed_invalid_exps = []
    for e in invalid_exps:
        config = e[29:-1]
        params = config.split(",", 9)
        parsed_config = dict()
        for param in params:
            param_name = param.split(":", 1)[0]
            param_value = param.split(":", 1)[1]
            if param_name == 'macs':
                param_value = param_value[:-2]
            if param_value.isdigit():
                param_value = int(param_value)
            parsed_config[param_name] = param_value
        parsed_invalid_exps.append(parsed_config)
    print(f"Discovered {len(parsed_invalid_exps)} invalid experiments.")
    return parsed_invalid_exps

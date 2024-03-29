import pandas as pd
from warnings import warn
import re


def parse_codecarbon_output(path, save_to_csv=True, slurm_log_info=None) -> pd.DataFrame:
    """
    takes the raw codecarbon output, parses the project_name column and returns a pandas DataFrame
    :param path: the path to the file that should be to be parsed
    :param slurm_log_info: a tuple of shape (path_to_file,data_collector_name,old_log_type,num_repeat_config) e.g. ('./slurm-111.out',conv2d,False,3)
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
    if slurm_log_info:
        if slurm_log_info[2]:
            invalid_configs = parse_old_slurm_log_for_errors(slurm_log_info[0], num_repeat_config=slurm_log_info[3])
        else:
            invalid_configs = parse_slurm_log_for_errors(slurm_log_info[0], num_repeat_config=slurm_log_info[3])
        print(f"Dropped observations with the following indices: {invalid_configs[slurm_log_info[1]]}")
        df = df.drop(index=invalid_configs[slurm_log_info[1]])
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
    if (df["cpu_energy"] < 0).any():
        previous_shape = df.shape
        df = df.loc[(df["cpu_energy"] > 0)]
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
    print(f"Final shape of data set: {df.shape}")
    return df


def parse_slurm_log_for_errors(path, num_repeat_config) -> []:
    """
    checks the slum cluster log for codecarbon errors and returns configuration indices which caused the errors
    :param num_repeat_config: the number of times each config was repeated in the measurement
    :param path: the path to the slurm output file
    :return:the invalid configurations indices (corresponding to rows in csv)
    """
    file = open(path, 'r')
    lines = file.readlines()
    curr_config_idx = 0
    invalid_configs = {}
    curr_data_collector = ""
    last_line = ""
    for line in lines:
        if line.startswith("Starting data-collection for"):
            curr_data_collector = line.split(" ")[-1][:-4]
            invalid_configs[curr_data_collector] = []
        elif line.endswith("% done)\n"):
            pattern_match = re.search("([0-9]+)\/([0-9]+)", line)
            curr_config_idx = int(pattern_match.groups()[0]) * num_repeat_config + int(pattern_match.groups()[1]) - 1
        elif line.startswith("[codecarbon ERROR"):
            if bool(re.search("codecarbon ERROR", last_line)):
                continue
            invalid_configs[curr_data_collector].append(curr_config_idx)
        last_line = line
    file.close()
    return invalid_configs


def parse_old_slurm_log_for_errors(path, num_repeat_config) -> []:
    """
    checks the slum cluster log for codecarbon errors and returns configuration indices which caused the errors
    :param num_repeat_config: the number of times each config was repeated in the measurement
    :param path: the path to the slurm output file
    :return:the invalid configurations indices (corresponding to rows in csv)
    """
    file = open(path, 'r')
    lines = file.readlines()
    invalid_configs_indices = {}
    current_config_indices = []
    current_data_collector = ""
    last_line = ""
    for line in lines:
        if line.startswith("Starting data-collection for"):
            current_data_collector = line.split(" ")[-1][:-4]
            invalid_configs_indices[current_data_collector] = []
        elif line.startswith("current config:"):
            pattern_match = re.search("([0-9]+)\/[0-9]+", line)
            base_idx = int(pattern_match.groups()[0]) * num_repeat_config
            current_config_indices = [base_idx + i for i in range(0, num_repeat_config)]
        elif bool(re.search("codecarbon ERROR", line)):
            if bool(re.search("codecarbon ERROR", last_line)) and len(invalid_configs_indices[current_data_collector]) > 0 and invalid_configs_indices[current_data_collector][-1] == current_config_indices[-1]:
                continue
            invalid_configs_indices[current_data_collector] += current_config_indices
        last_line = line
    file.close()
    return invalid_configs_indices

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
    # module:Conv2d,rep_no:1,macs:925646848,batch_size:1,image_size:56,kernel_size:3,in_channels:128,out_channels:256,stride:1,padding:1,note:vgg16(layer_idx:10),forward_passes:1295
    # count the number of properties saved in the 'project_name' column
    col_names = [s.split(':', 1)[0] for s in df.project_name.iloc[0].split(',')]
    df[col_names] = df["project_name"].str.split(pat=",", n=len(col_names) - 1, expand=True)
    for col in col_names:
        df[col] = df[col].str[df[col].iloc[0].find(":") + 1:]
        if df[col].iloc[0].isnumeric():
            df[col] = pd.to_numeric(df[col])
    # parse note column
    architecture_col = []
    layer_indices_col = []
    for idx, row in df.iterrows():
        if row.note != 'empty':
            match = re.search(r"([a-z,0-9]+)\(.*:([0-9]+)\)", row.note)
            if match:
                architecture_col.append(match.groups()[0])
                layer_indices_col.append(match.groups()[1])
            else:
                warn("Something went wrong while parsing the 'note' column")
        else:
            architecture_col.append(None)
            layer_indices_col.append(None)
    df['architecture'] = architecture_col
    df['layer_idx'] = layer_indices_col
    df.drop(['project_name', 'note'], axis=1, inplace=True)
    if save_to_csv:
        df.to_csv(re.sub(r'(.+)(-raw)(.csv)', r'\1' + '-parsed' + r'\3', path))
    return df


def preprocess_and_normalize_energy_data(df, param_cols, aggregate=True) -> pd.DataFrame:
    """
    this function normalizes the measured energy_values by the number of forward-passes and aggregates repeated configs
    :param df: the pd.DataFrame containing the data from the parsed codecarbon output
    :param param_cols: the parameter names of the module configuration
    :param aggregate: whether to compute the mean-energy of configurations that are identical
    :return: the preprocessed pd.DataFrame
    """
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
        print(
            f"Shape before aggregation: {previous_shape}, after aggregation: {df.shape} (non numeric columns removed)")
    return df

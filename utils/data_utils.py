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
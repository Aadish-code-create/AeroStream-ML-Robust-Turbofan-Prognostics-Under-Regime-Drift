import pandas as pd

columns = (
    ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss_file(file_number):
    file_path = f"data/CMAPSS/train_FD00{file_number}.txt"

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    # Remove the two trailing empty columns
    df = df.iloc[:, :26]

    df.columns = columns

    print(f"Loaded FD00{file_number}: {df.shape}")

    return df


train_datasets = {
    f"FD00{i}": load_cmapss_file(i)
    for i in range(1, 5)
}

print(train_datasets["FD001"].head())
from pathlib import Path

import pandas as pd

# Load the label table relative to the project root.
labels_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "secom_labels.data"
df_labels = pd.read_csv(
    labels_path, header=0, sep="\t"
)

print("Labels shape:", df_labels.shape)
print(df_labels.to_string(index=False))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 0.0  & x13                          & 0.1454 & 0.3150 \\
# 1.1  & segmenter\_x13\_x13          & 0.4639 & 0.7424 \\
# 2.1  & swint\_x13\_x13              & 0.8103 & 1.1340 \\
# 2.2  & swint\_x13\_x13\_v2          & 0.2155 & 0.3634 \\
# 2.4  & swint\_x13\_x13\_v4          & 0.3208 & 0.6189 \\
# 2.5  & swint\_x13\_x13\_v5          & 0.1725 & 0.3638 \\
# 2.6  & swint\_x13\_x13\_v6          & 0.2984 & 0.6011 \\
# 3.1  & swint\_att\_x13              & 0.8084 & 0.9432 \\
# 3.2  & x13t                         & 0.2185 & 0.4036 \\
# 4.1  & x13\_x13\_swint              & 0.1175 & 0.3401 \\
# 4.2  & x13\_x13\_swintv2            & 0.1695 & 0.3226 \\
# 4.3  & x13\_x13\_swintv3            & 0.1158 & 0.3164 \\
# 4.4  & x13\_x13\_swintv4            & 0.1354 & 0.3255 \\
# 4.5  & x13\_x13\_swintv5            & 0.1461 & 0.3268 \\
# 5.1  & xtranfuser                   & 0.5532 & 0.7135 \\
# 5.2  & xtranfuserv2                 & 0.5103 & 0.6948 \\
# 5.3  & xtranfuserv3                 & 0.3090 & 0.5925 \\
# 5.4  & xtranfuserv4                 & 0.1637 & 0.3183 \\
# 6.1  & x13\_x13\_convnext           & 0.1022 & 0.2997 \\
# 7.1  & x13\_x13\_swin\_torch        & 0.1539 & 0.2992 \\
# 7.2  & x13\_x13\_swin\_torchv2      & 0.1552 & 0.3061 \\
# 8.1  & x13\_x13\_swin\_torch\_16    & 0.6141 & 0.7981 \\
# Data dari tabel 4.4 (Train Loss dan Val Loss)
data = {
    "Model": [
        "x13",
        "xsegmenter",
        "SKGE-Swin (stage 4→1)",
        "SKGE-Swin (stage 1→4)",
        "SKGE-Swin (stage 1, 2, 3→4)",
        "SKGE-Swin (stage 3)",
        "SKGE-Swin (stage 2→3)",
        "cx13",
        "x13t",
        "SKGE-Swin (stage 3)",
        "SKGE-Swin (stage 2→4)",
        "SKGE-Swin (stage 1→4)",
        "SKGE-Swin (stage 1→4) + lidar",
        "SKGE-Swin-tiny (stage 4)",
        "xtransfuser",
        "xtransfuserv2",
        "xtransfuserv2 (transfuser)",
        "xtransfuserv2 (cognitiver transfuser)",
        "xconvnext",
        "SKGE-Swin-tiny (stage 4)",
        "SKGE-Swin-tiny (stage 1→4)",
        "SKGE-Swin-tiny (stage 4) float16",
        "SKGE-Swin-tiny (stage 1→4) float16"
    ],
    "Train Loss": [
        0.1454, 0.4639, 0.8103, 0.2155, 0.3208, 0.1725, 0.2984, 0.8084, 0.2185, 0.1175, 0.1695, 0.1158, 0.1354, 0.1461, 0.5532, 0.5103, 0.3090, 0.1637, 0.1022, 0.1539, 0.1552, 0.6141, 1.5716],
    "Val Loss": [
        0.3150, 0.7424, 1.1340, 0.3634, 0.6189, 0.3638, 0.6011, 0.9432, 0.4036, 0.3401, 0.3226, 0.3164, 0.3255, 0.3268, 0.7135, 0.6948, 0.5925, 0.3183, 0.2997, 0.2992, 0.3061, 0.7981, 1.6783]
}

# Buat DataFrame
df = pd.DataFrame(data)
# Tentukan batas minimum dan maksimum dari kedua kolom
min_val = min(df["Train Loss"].min(), df["Val Loss"].min())
max_val = max(df["Train Loss"].max(), df["Val Loss"].max())

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Train Loss", y="Val Loss", data=df,
                hue="Model", palette="tab20", s=100)

# Tambahkan ideal line (y = x)
plt.plot([min_val, max_val], [min_val, max_val],
         'r--', label='Ideal Line (Train = Val)')

# Label dan Judul
plt.title("Scatter Plot Train Loss vs Validation Loss per Model", fontsize=14)
plt.xlabel("Train Loss", fontsize=12)
plt.ylabel("Validation Loss", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Tampilkan plot
plt.show()

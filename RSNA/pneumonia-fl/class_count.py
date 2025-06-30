import pandas as pd

df = pd.read_csv("D:/Ascl_Mimic_Data/RSNA/stage2_train_metadata.csv")
class_counts = df['Target'].value_counts()

num_class_0 = class_counts.get(0, 0)
num_class_1 = class_counts.get(1, 0)

print(f"Class 0: {num_class_0}")
print(f"Class 1: {num_class_1}")

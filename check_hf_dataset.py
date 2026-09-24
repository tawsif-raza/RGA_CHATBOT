from datasets import load_dataset

# 1. Load the dataset from your local JSONL file
# Replace "your_dataset.jsonl" with your actual file name if it's different
dataset = load_dataset("json", data_files="your_dataset.jsonl")

print("=" * 50)
print("📊 HUGGING FACE DATASET OVERVIEW")
print("=" * 50)
# This prints the structural split (e.g., 'train'), row count, and column names
print(dataset)

print("\n" + "=" * 50)
print("⚙️ DATASET PARAMETERS / FEATURES")
print("=" * 50)
# This displays data types (Features) for each column/parameter
print(dataset["train"].features)

print("\n" + "=" * 50)
print("📏 DATASET SIZE")
print("=" * 50)
print(f"Total Rows (Samples): {dataset['train'].num_rows}")
print(f"Total Columns (Parameters): {dataset['train'].num_columns}")

print("\n" + "=" * 50)
print("👀 FIRST SAMPLE PREVIEW")
print("=" * 50)
# This displays the structure of the very first data sample
print(dataset["train"][0])

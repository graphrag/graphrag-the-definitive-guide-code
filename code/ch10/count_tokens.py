import csv
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

CSV_PATH = "../../data/ch10/articles_initial.csv"

token_counts = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        token_counts.append(len(enc.encode(row["text"])))

# tag::stats[]
counts = np.array(token_counts)

print(f"Rows:  {len(counts)}")
print(f"Min:   {int(np.min(counts))}")
print(f"P25:   {int(np.percentile(counts, 25))}")
print(f"P50:   {int(np.percentile(counts, 50))}")
print(f"P75:   {int(np.percentile(counts, 75))}")
print(f"Max:   {int(np.max(counts))}")
# end::stats[]

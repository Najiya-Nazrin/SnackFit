import pickle

with open("data/meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Print the type and length
print(f"Type: {type(meta)}")
print(f"Number of entries: {len(meta)}")

# Print first few entries nicely
for i, item in enumerate(meta[:5]):
    print(f"Entry {i+1}:")
    for key, value in item.items():
        print(f"  {key}: {value}")
    print()

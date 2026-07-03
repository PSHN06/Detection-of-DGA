import pandas as pd
import numpy as np
import random
import datetime

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Date range of Dataset A: 2026-01-01 to 2026-03-31
start_date = datetime.date(2026, 1, 1)
end_date = datetime.date(2026, 3, 31)
date_range_days = (end_date - start_date).days + 1

def random_date():
    random_days = random.randint(0, date_range_days - 1)
    return (start_date + datetime.timedelta(days=random_days)).strftime("%Y-%m-%d")

# Malicious generation: random consonant clusters
consonants = "bcdfghjklmnpqrstvwxyz"
malicious_domains = []
tlds = [".com", ".net", ".org"]

for _ in range(50000):
    length = random.randint(5, 9)
    name = "".join(random.choice(consonants) for _ in range(length))
    tld = random.choice(tlds)
    domain = name + tld
    malicious_domains.append({
        'domain': domain,
        'timestamp': random_date(),
        'family': 'short_dga',
        'class': 1
    })

# Benign generation: alternating pronounceable syllables
vowels = "aeiou"
benign_domains = []

for _ in range(50000):
    length = random.randint(5, 9)
    name_chars = []
    # alternate consonant and vowel
    start_with_consonant = random.choice([True, False])
    for idx in range(length):
        if (idx % 2 == 0) == start_with_consonant:
            name_chars.append(random.choice(consonants))
        else:
            name_chars.append(random.choice(vowels))
    name = "".join(name_chars)
    tld = random.choice(tlds)
    domain = name + tld
    benign_domains.append({
        'domain': domain,
        'timestamp': random_date(),
        'family': 'short_benign',
        'class': 0
    })

# Load original files (fresh copy of the original 184,410 datasets to avoid compounding)
# Wait, we want to load a fresh copy of the original datasets.
# Since we overwrote malicious_raw.csv and benign_raw.csv in the previous step,
# they now have 194,410 rows. We can read them, but we should drop the previous 'short_dga' and 'short_benign'
# families first so we don't duplicate them!
print("Loading current malicious_raw.csv and benign_raw.csv...")
mal_raw = pd.read_csv("malicious_raw.csv")
ben_raw = pd.read_csv("benign_raw.csv")

# Drop any previous short injections
mal_raw = mal_raw[mal_raw['family'] != 'short_dga']
ben_raw = ben_raw[ben_raw['family'] != 'short_benign']

# Create DataFrames
mal_new = pd.DataFrame(malicious_domains)
ben_new = pd.DataFrame(benign_domains)

# Concatenate
mal_updated = pd.concat([mal_raw, mal_new], ignore_index=True)
ben_updated = pd.concat([ben_raw, ben_new], ignore_index=True)

# Save
print(f"Cleaned base malicious count: {len(mal_raw)}, New count: {len(mal_updated)}")
print(f"Cleaned base benign count: {len(ben_raw)}, New count: {len(ben_updated)}")
mal_updated.to_csv("malicious_raw.csv", index=False)
ben_updated.to_csv("benign_raw.csv", index=False)
print("Updated raw datasets saved successfully.")

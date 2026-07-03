import os
import pandas as pd
import subprocess

def main():
    mal_file = "malicious_raw.csv"
    ben_file = "benign_raw.csv"
    
    print("Loading raw files for duplicate audit...")
    mal_df = pd.read_csv(mal_file)
    ben_df = pd.read_csv(ben_file)
    
    print(f"Initial raw rows:")
    print(f"  - Malicious: {len(mal_df)}")
    print(f"  - Benign:    {len(ben_df)}")
    
    # Check for duplicate domains
    mal_dups = mal_df.duplicated(subset=['domain']).sum()
    ben_dups = ben_df.duplicated(subset=['domain']).sum()
    
    print(f"Found duplicates:")
    print(f"  - Malicious duplicates: {mal_dups}")
    print(f"  - Benign duplicates:    {ben_dups}")
    
    # Check if there are domains that exist in both malicious and benign sets
    overlap = set(mal_df['domain']).intersection(set(ben_df['domain']))
    print(f"Overlap (domains in both sets): {len(overlap)}")
    
    # Drop duplicates
    if mal_dups > 0:
        print("Dropping duplicates from malicious raw...")
        mal_df = mal_df.drop_duplicates(subset=['domain'])
        mal_df.to_csv(mal_file, index=False)
        
    if ben_dups > 0:
        print("Dropping duplicates from benign raw...")
        ben_df = ben_df.drop_duplicates(subset=['domain'])
        ben_df.to_csv(ben_file, index=False)
        
    if len(overlap) > 0:
        print("Removing overlapping domains from both sets to prevent label confusion...")
        mal_df = mal_df[~mal_df['domain'].isin(overlap)]
        ben_df = ben_df[~ben_df['domain'].isin(overlap)]
        mal_df.to_csv(mal_file, index=False)
        ben_df.to_csv(ben_file, index=False)
        
    print(f"Audited raw rows after deduplication:")
    print(f"  - Malicious: {len(mal_df)}")
    print(f"  - Benign:    {len(ben_df)}")
    print("Raw files deduplicated and saved successfully.")
    
    # Re-run temporal splitting script to rebuild train, val, and test partitions
    print("\nRe-running chronological temporal splitting script...")
    result = subprocess.run(
        ["python", "step3_temporal_split.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error re-running temporal split:")
        print(result.stderr)
        
if __name__ == "__main__":
    main()

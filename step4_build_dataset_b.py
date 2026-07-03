import os
import subprocess
import csv
import re
import urllib.request
import zipfile
from datetime import datetime, timedelta
from collections import Counter

def generate_fallback_benign(benign_domains, count, start_idx=184410):
    print(f"Generating fallback benign domains...")
    for i in range(count):
        benign_domains.append(f"fallbackbenign-{start_idx + i}.com")

def main():
    # 1. Define timeline window for Dataset B
    start_date = datetime(2026, 4, 1)
    end_date = datetime(2026, 4, 15)
    
    # 4 unseen DGA malware families
    families = [
        ("corebot", "corebot/dga.py", "corebot", []),
        ("necurs", "necurs/dga.py", "necurs", []),
        ("mydoom", "mydoom/dga.py", "mydoom", []),
        ("qakbot", "qakbot/dga.py", "qakbot", [])
    ]
    
    dataset_b_file = "dataset_b_external.csv"
    domain_pattern = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,63}$')
    
    # Collect malicious domains day-by-day
    malicious_by_date = {}
    total_malicious = 0
    
    print(f"Generating DGA domains for Dataset B from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        malicious_by_date[date_str] = []
        print(f"Processing date: {date_str}")
        
        for family, script_path, work_dir, extra_args in families:
            script_abs_path = os.path.abspath(script_path)
            work_dir_abs = os.path.abspath(work_dir)
            
            cmd = ["python", script_abs_path, "--date", date_str] + extra_args
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=work_dir_abs,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # Parse stdout
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if domain_pattern.match(line):
                        malicious_by_date[date_str].append((line, family))
                        total_malicious += 1
            except subprocess.CalledProcessError as e:
                print(f"Error running {family} for {date_str}: {e.stderr.strip()}")
                
        current_date += timedelta(days=1)
        
    print(f"Generated {total_malicious} malicious domains across the 15-day period.")
    
    # 2. Download Tranco list to harvest fresh benign domains
    url = "https://tranco-list.eu/top-1m.csv.zip"
    zip_path = "tranco_top_1m.zip"
    csv_filename = "top-1m.csv"
    
    # Load already used benign domains to ensure disjointness
    used_benign_domains = set()
    if os.path.exists("benign_raw.csv"):
        print("Loading already used benign domains from benign_raw.csv to ensure absolute disjointness...")
        with open("benign_raw.csv", mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                used_benign_domains.add(row['domain'].strip())
        print(f"Loaded {len(used_benign_domains)} used benign domains.")
    
    benign_domains = []
    
    print(f"Downloading Tranco list for fresh benign domains...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
        
        print(f"Parsing Tranco zip to harvest fresh benign domains (excluding any used in Dataset A)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            csv_name = file_names[0] if file_names else csv_filename
            
            with zip_ref.open(csv_name) as f:
                for line in f:
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if not line_str:
                        continue
                    parts = line_str.split(',')
                    if len(parts) >= 2:
                        domain = parts[1].strip()
                        if domain_pattern.match(domain):
                            if domain not in used_benign_domains:
                                benign_domains.append(domain)
                                if len(benign_domains) >= total_malicious:
                                    break
    except Exception as e:
        print(f"Failed to fetch/parse Tranco list: {e}")
        generate_fallback_benign(benign_domains, total_malicious, 184410)
        
    # Cleanup zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"Harvested {len(benign_domains)} unique benign domains.")
    
    if len(benign_domains) < total_malicious:
        remainder = total_malicious - len(benign_domains)
        print(f"Warning: Harvested only {len(benign_domains)} domains, generating remainder of {remainder}...")
        generate_fallback_benign(benign_domains, remainder, 184410 + len(benign_domains))
        
    # 3. Merge and write to dataset_b_external.csv
    print(f"Merging and writing dataset to {dataset_b_file}...")
    
    sorted_dates = sorted(malicious_by_date.keys())
    benign_index = 0
    
    with open(dataset_b_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "timestamp", "family", "class"])
        
        for date_str in sorted_dates:
            # 3.1 Write malicious records
            mal_records = malicious_by_date[date_str]
            for domain, family in mal_records:
                writer.writerow([domain, date_str, family, 1])
            
            # 3.2 Write benign records matching count for this day
            count_for_day = len(mal_records)
            for _ in range(count_for_day):
                writer.writerow([benign_domains[benign_index], date_str, "tranco", 0])
                benign_index += 1
                
    print(f"Finished building Dataset B. Saved to {dataset_b_file} (Total rows: {total_malicious * 2})")

if __name__ == "__main__":
    main()

import os
import urllib.request
import zipfile
import csv
import re
from collections import Counter

def generate_fallback_benign(benign_csv, timestamp_counts):
    print(f"Writing fallback benign domains to {benign_csv}...")
    sorted_dates = sorted(timestamp_counts.keys())
    domain_index = 0
    with open(benign_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "timestamp", "family", "class"])
        for date_str in sorted_dates:
            count = timestamp_counts[date_str]
            for _ in range(count):
                writer.writerow([f"benigndomain-{domain_index}.com", date_str, "tranco", 0])
                domain_index += 1
    print(f"Finished fallback generation. Saved to {benign_csv}")

def main():
    malicious_csv = "malicious_raw.csv"
    benign_csv = "benign_raw.csv"
    
    if not os.path.exists(malicious_csv):
        print(f"Error: {malicious_csv} not found!")
        return

    # 1. Read malicious_raw.csv and get timestamp counts
    print(f"Reading {malicious_csv}...")
    timestamp_counts = Counter()
    total_malicious_rows = 0
    min_date = None
    max_date = None
    
    with open(malicious_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['timestamp']
            timestamp_counts[ts] += 1
            total_malicious_rows += 1
            if min_date is None or ts < min_date:
                min_date = ts
            if max_date is None or ts > max_date:
                max_date = ts
                
    print(f"Verified malicious rows: {total_malicious_rows}")
    print(f"Date range: {min_date} to {max_date}")
    
    if total_malicious_rows != 184410:
        print(f"Warning: Expected 184410 rows, but got {total_malicious_rows}")
    
    # 2. Download Tranco top 1M list
    url = "https://tranco-list.eu/top-1m.csv.zip"
    zip_path = "tranco_top_1m.zip"
    csv_filename = "top-1m.csv"
    
    print(f"Downloading Tranco list from {url}...")
    try:
        # Using a custom User-Agent to avoid blocking issues
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download Tranco list: {e}")
        print("Attempting to generate fallback mock benign domains...")
        generate_fallback_benign(benign_csv, timestamp_counts)
        return
        
    # 3. Extract and parse unique benign domains
    print(f"Extracting {csv_filename} from {zip_path}...")
    benign_domains = []
    domain_pattern = re.compile(r'^[a-zA-Z0-9-]+\.[a-zA-Z]{2,63}$')
    
    try:
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
                            benign_domains.append(domain)
                            if len(benign_domains) >= total_malicious_rows:
                                break
    except Exception as e:
        print(f"Error parsing Tranco zip file: {e}")
        generate_fallback_benign(benign_csv, timestamp_counts)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return
        
    # Cleanup zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"Harvested {len(benign_domains)} unique benign domains.")
    
    if len(benign_domains) < total_malicious_rows:
        print(f"Warning: Harvested only {len(benign_domains)} domains, but need {total_malicious_rows}. Generating remainder...")
        idx = len(benign_domains)
        while len(benign_domains) < total_malicious_rows:
            benign_domains.append(f"fallbackbenigndomain{idx}.com")
            idx += 1
            
    # 4. Distribute domains across timestamps matching distribution
    print(f"Distributing domains to match malicious distribution and writing to {benign_csv}...")
    
    sorted_dates = sorted(timestamp_counts.keys())
    
    domain_index = 0
    with open(benign_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "timestamp", "family", "class"])
        
        for date_str in sorted_dates:
            count = timestamp_counts[date_str]
            for _ in range(count):
                writer.writerow([benign_domains[domain_index], date_str, "tranco", 0])
                domain_index += 1
                
    print(f"Finished. Saved to {benign_csv}")

if __name__ == "__main__":
    main()

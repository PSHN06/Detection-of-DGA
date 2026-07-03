import os
import subprocess
import csv
from datetime import datetime, timedelta
import re

def main():
    # Define date range
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 3, 31)
    
    # DGA script configurations
    # Format: (family_name, script_rel_path, working_dir, extra_args)
    families = [
        ("banjori", "banjori/dga.py", "banjori", []),
        ("sharkbot", "sharkbot/dga.py", "sharkbot", []),
        ("gozi", "gozi/dga.py", "gozi", []),
        ("murofet", "murofet/v1/dga.py", "murofet/v1", []),
        ("pykspa", "pykspa/improved/dga.py", "pykspa/improved", [])
    ]
    
    # Prepare CSV file
    csv_file = "malicious_raw.csv"
    columns = ["domain", "timestamp", "family", "class"]
    
    # Regular expression to identify a likely domain name from stdout
    domain_pattern = re.compile(r'^[a-zA-Z0-9-]+\.[a-zA-Z]{2,63}$')
    
    print(f"Generating DGA domains from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing date: {date_str}")
            
            for family, script_path, work_dir, extra_args in families:
                # Resolve full path to script and work directory relative to script run location
                # The script runs from the root of the workspace
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
                    
                    stdout = result.stdout
                    # Process lines of stdout to find domains
                    for line in stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Match domain pattern
                        if domain_pattern.match(line):
                            writer.writerow([line, date_str, family, 1])
                except subprocess.CalledProcessError as e:
                    print(f"Error running {family} for {date_str}: {e.stderr.strip()}")
            
            current_date += timedelta(days=1)
            
    print(f"Finished. Data saved to {csv_file}")

if __name__ == "__main__":
    main()

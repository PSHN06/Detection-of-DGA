import os
import sys
import time
import socket
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Try loading scapy
try:
    from scapy.all import IP, UDP, DNS, DNSQR, send
    has_scapy = True
except ImportError:
    has_scapy = False

def send_scapy_dns(domain, dns_server="8.8.8.8"):
    try:
        pkt = IP(dst=dns_server)/UDP(sport=np.random.randint(1024, 65535), dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))
        send(pkt, verbose=0)
        return True
    except Exception:
        return False

def send_socket_dns(domain):
    try:
        # Standard socket resolution (non-blocking thread)
        socket.gethostbyname(domain)
    except Exception:
        # Most DGA domains will fail to resolve, which is expected
        pass

def main():
    print("======================================================================")
    print("                   DGA NETWORK ATTACK EMULATOR ACTIVE                  ")
    print("======================================================================")
    
    if not os.path.exists("test.csv"):
        print("Error: test.csv not found! Run the pipeline splits first.")
        return
        
    print("Loading test dataset domains...")
    test_df = pd.read_csv("test.csv")
    
    # Separate benign and malicious for structured emulations
    benign_domains = test_df[test_df['class'] == 0]['domain'].tolist()
    dga_domains = test_df[test_df['class'] == 1]['domain'].tolist()
    
    print(f"Loaded {len(benign_domains)} benign domains and {len(dga_domains)} DGA domains.")
    print("Beginning active DGA attack emulation loop...")
    print("Press Ctrl+C to terminate.")
    
    # We will use ThreadPoolExecutor to prevent blocking on DNS resolution timeouts
    executor = ThreadPoolExecutor(max_workers=5)
    
    packet_count = 0
    scapy_working = has_scapy
    
    while True:
        try:
            # Emulate structured traffic: 80% benign queries, 20% DGA attacks
            is_attack = np.random.rand() < 0.20
            
            if is_attack and len(dga_domains) > 0:
                domain = np.random.choice(dga_domains)
                query_type = "[ATTACK DGA]"
            else:
                domain = np.random.choice(benign_domains)
                query_type = "[BENIGN]"
                
            packet_count += 1
            print(f"[{time.strftime('%H:%M:%S')}] Emulator Event #{packet_count:<4} -> Querying {query_type:<12} domain: {domain}")
            
            # Send query
            if scapy_working:
                success = send_scapy_dns(domain)
                if not success:
                    scapy_working = False
                    executor.submit(send_socket_dns, domain)
            else:
                executor.submit(send_socket_dns, domain)
                
            # Random interval between events to simulate real network query rates
            time.sleep(np.random.uniform(0.5, 2.0))
            
        except KeyboardInterrupt:
            print("\nExiting attack emulator.")
            break

if __name__ == "__main__":
    main()

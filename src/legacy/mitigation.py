import time
import threading
from collections import defaultdict
from src.config import CONFIDENCE_THRESHOLD, RATE_LIMIT_DEFAULT, RATE_LIMIT_ATTACK, MITIGATION_DURATION

class RateLimiter:
    def __init__(self, default_limit=RATE_LIMIT_DEFAULT, attack_limit=RATE_LIMIT_ATTACK, duration=MITIGATION_DURATION):
        self.default_limit = default_limit
        self.attack_limit = attack_limit
        self.duration = duration
        self.request_counts = defaultdict(list)
        self.mitigated_ips = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, ip_address, confidence_score):
        with self.lock:
            current_time = time.time()
            
            if ip_address in self.mitigated_ips:
                mitigation_end = self.mitigated_ips[ip_address]
                if current_time < mitigation_end:
                    return False
                else:
                    del self.mitigated_ips[ip_address]
            
            if confidence_score >= CONFIDENCE_THRESHOLD:
                self.mitigated_ips[ip_address] = current_time + self.duration
                return False
            
            current_limit = self.default_limit if confidence_score < 0.5 else self.attack_limit
            
            request_times = self.request_counts[ip_address]
            request_times = [t for t in request_times if current_time - t < 1.0]
            self.request_counts[ip_address] = request_times
            
            if len(request_times) >= current_limit:
                return False
            
            self.request_counts[ip_address].append(current_time)
            return True
    
    def print_status(self):
        with self.lock:
            print(f"\nRate Limiter Status:")
            print(f"  Active mitigations: {len(self.mitigated_ips)} IPs")
            print(f"  Active request counts: {len(self.request_counts)} IPs")

def dynamic_rate_limit(ip_address, confidence_score):
    print(f"Request from {ip_address} - Confidence: {confidence_score:.3f}")
    
    if confidence_score >= CONFIDENCE_THRESHOLD:
        print(f"  ATTACK DETECTED! Mitigating IP {ip_address}")
        return False
    elif confidence_score >= 0.6:
        print(f"  Suspicious traffic from {ip_address} - applying rate limit")
        return True
    else:
        print(f"  Normal traffic from {ip_address} - allowing")
        return True

def block_ip(ip_address, duration_seconds):
    print(f"BLOCKING IP {ip_address} for {duration_seconds} seconds")
    return True

def unblock_ip(ip_address):
    print(f"UNBLOCKING IP {ip_address}")
    return True

class TrafficFilter:
    def __init__(self):
        self.blocked_ips = {}
        self.suspicious_ips = defaultdict(int)
    
    def filter_packet(self, ip_address, packet_info):
        current_time = time.time()
        
        if ip_address in self.blocked_ips:
            if current_time < self.blocked_ips[ip_address]:
                return False
            else:
                del self.blocked_ips[ip_address]
        
        return True
    
    def mark_suspicious(self, ip_address):
        self.suspicious_ips[ip_address] += 1
        
        if self.suspicious_ips[ip_address] >= 10:
            self.blocked_ips[ip_address] = time.time() + MITIGATION_DURATION
            print(f"IP {ip_address} blocked due to repeated suspicious activity")
    
    def clear_expired(self):
        current_time = time.time()
        expired = [ip for ip, expiry in self.blocked_ips.items() if current_time >= expiry]
        for ip in expired:
            del self.blocked_ips[ip]
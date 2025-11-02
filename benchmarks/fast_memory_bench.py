#!/usr/bin/env python3
"""
Fast Memory Benchmark for Hackathon Demo
Tests memory allocation and I/O operations in ~8 seconds
"""

import time
import random
import sys
import tempfile
import os

def memory_allocation_test(iterations=1000):
    """Test memory allocation speed"""
    start = time.time()
    allocations = []
    
    for i in range(iterations):
        # Allocate 1MB chunks
        data = bytearray(1024 * 1024)
        # Write some data to ensure it's actually allocated
        data[0] = i % 256
        data[-1] = (i * 2) % 256
        allocations.append(data)
        
        # Clear some old allocations to avoid OOM
        if len(allocations) > 50:
            allocations.pop(0)
    
    elapsed = time.time() - start
    return iterations / elapsed

def file_io_test(num_files=100, file_size_kb=100):
    """Test file I/O with dirty page flushing"""
    start = time.time()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Write test
        for i in range(num_files):
            filepath = os.path.join(temp_dir, f'test_{i}.dat')
            with open(filepath, 'wb') as f:
                # Write random data
                data = bytearray(random.getrandbits(8) for _ in range(file_size_kb * 1024))
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        
        # Read test
        for i in range(num_files):
            filepath = os.path.join(temp_dir, f'test_{i}.dat')
            with open(filepath, 'rb') as f:
                data = f.read()
        
        elapsed = time.time() - start
        total_ops = num_files * 2  # writes + reads
        return total_ops / elapsed
        
    finally:
        # Cleanup
        for i in range(num_files):
            try:
                os.unlink(os.path.join(temp_dir, f'test_{i}.dat'))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

def main():
    """Run fast memory + I/O benchmark"""
    print("Starting fast memory benchmark...", file=sys.stderr)
    
    # Test 1: Memory allocation (lighter test)
    print("Running memory allocation test...", file=sys.stderr)
    mem_ops = memory_allocation_test(iterations=500)
    
    # Test 2: File I/O (tests dirty page handling)
    print("Running file I/O test...", file=sys.stderr)
    io_ops = file_io_test(num_files=50, file_size_kb=50)
    
    # Combined score
    total_ops = mem_ops + io_ops
    
    # Output in parseable format
    print(f"Memory Operations/sec: {mem_ops:.2f}")
    print(f"I/O Operations/sec: {io_ops:.2f}")
    print(f"operations_per_second: {total_ops:.2f}")
    print(f"Combined Score: {total_ops:.2f}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

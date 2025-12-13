#!/usr/bin/env python3
"""
High-speed benchmark sender for fraud detection pipeline.
Sends all test data as fast as possible.
"""

import asyncio
import json
import time
import pandas as pd
import nats
from datetime import datetime

async def benchmark_send(nats_url: str, subject: str, data_path: str, limit: int = None):
    """Send test transactions as fast as possible."""
    
    print(f"Connecting to NATS at {nats_url}...")
    nc = await nats.connect(nats_url)
    print(f"Connected! Publishing to subject: {subject}")
    
    # Load test data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if limit:
        df = df.head(limit)
    
    total = len(df)
    defaults = df['is_default'].sum()
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Sending {total} transactions")
    print(f"Defaults in data: {defaults} ({defaults/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Prepare all messages first
    messages = []
    for idx, row in df.iterrows():
        transaction = {
            "transaction_id": str(int(row['ID'])),
            "limit_bal": float(row['LIMIT_BAL']),
            "sex": int(row['SEX']),
            "education": int(row['EDUCATION']),
            "marriage": int(row['MARRIAGE']),
            "age": int(row['AGE']),
            "pay_0": int(row['PAY_0']),
            "pay_2": int(row['PAY_2']),
            "pay_3": int(row['PAY_3']),
            "pay_4": int(row['PAY_4']),
            "pay_5": int(row['PAY_5']),
            "pay_6": int(row['PAY_6']),
            "bill_amt1": float(row['BILL_AMT1']),
            "bill_amt2": float(row['BILL_AMT2']),
            "bill_amt3": float(row['BILL_AMT3']),
            "bill_amt4": float(row['BILL_AMT4']),
            "bill_amt5": float(row['BILL_AMT5']),
            "bill_amt6": float(row['BILL_AMT6']),
            "pay_amt1": float(row['PAY_AMT1']),
            "pay_amt2": float(row['PAY_AMT2']),
            "pay_amt3": float(row['PAY_AMT3']),
            "pay_amt4": float(row['PAY_AMT4']),
            "pay_amt5": float(row['PAY_AMT5']),
            "pay_amt6": float(row['PAY_AMT6']),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        messages.append(json.dumps(transaction).encode())
    
    print(f"Messages prepared. Starting send...")
    
    # Send as fast as possible
    start_time = time.time()
    
    for i, payload in enumerate(messages):
        await nc.publish(subject, payload)
        
        # Progress every 1000
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Sent {i+1}/{total} ({rate:.0f} msg/s)")
    
    # Flush to ensure all sent
    await nc.flush()
    
    end_time = time.time()
    duration = end_time - start_time
    rate = total / duration
    
    await nc.drain()
    
    print(f"\n{'='*60}")
    print(f"SEND COMPLETE")
    print(f"{'='*60}")
    print(f"  Total sent: {total}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Send rate: {rate:.0f} messages/second")
    print(f"{'='*60}\n")
    
    return total, duration

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nats-url', default='nats://localhost:4222')
    parser.add_argument('--subject', default='transactions')
    parser.add_argument('--data', default='training/data/test.parquet')
    parser.add_argument('--limit', type=int, default=None, help='Limit records (default: all)')
    args = parser.parse_args()
    
    asyncio.run(benchmark_send(args.nats_url, args.subject, args.data, args.limit))


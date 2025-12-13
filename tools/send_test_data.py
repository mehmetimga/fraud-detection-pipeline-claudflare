#!/usr/bin/env python3
"""
Send test data from Parquet file to NATS message broker.
"""

import asyncio
import json
import argparse
import pandas as pd
from pathlib import Path
import nats
from datetime import datetime

async def send_test_data(
    nats_url: str,
    subject: str,
    data_path: str,
    limit: int = None,
    delay_ms: int = 100
):
    """Send test transactions to NATS."""
    
    print(f"Connecting to NATS at {nats_url}...")
    nc = await nats.connect(nats_url)
    print(f"Connected! Publishing to subject: {subject}")
    
    # Load test data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if limit:
        df = df.head(limit)
    
    print(f"Sending {len(df)} transactions...")
    
    sent = 0
    defaults = 0
    
    for idx, row in df.iterrows():
        # Create transaction JSON
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
        
        # Track actual defaults
        if row['is_default'] == 1:
            defaults += 1
        
        # Publish to NATS
        payload = json.dumps(transaction).encode()
        await nc.publish(subject, payload)
        sent += 1
        
        if sent % 100 == 0:
            print(f"Sent {sent}/{len(df)} transactions ({defaults} actual defaults)")
        
        await asyncio.sleep(delay_ms / 1000.0)
    
    await nc.drain()
    print(f"\nDone! Sent {sent} transactions ({defaults} actual defaults, {defaults/sent*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Send test data to NATS')
    parser.add_argument('--nats-url', default='nats://localhost:4222', help='NATS server URL')
    parser.add_argument('--subject', default='transactions', help='NATS subject')
    parser.add_argument('--data', default='training/data/test.parquet', help='Path to test data')
    parser.add_argument('--limit', type=int, default=100, help='Number of transactions to send')
    parser.add_argument('--delay', type=int, default=50, help='Delay between messages (ms)')
    args = parser.parse_args()
    
    asyncio.run(send_test_data(
        args.nats_url,
        args.subject,
        args.data,
        args.limit,
        args.delay
    ))

if __name__ == '__main__':
    main()


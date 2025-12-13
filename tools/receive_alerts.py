#!/usr/bin/env python3
"""
Receive and log fraud alerts from NATS message broker.
"""

import asyncio
import json
import argparse
from datetime import datetime
import nats

# Store received alerts for analysis
received_alerts = []

async def receive_alerts(
    nats_url: str,
    subject: str,
    timeout: int = 60
):
    """Receive alerts from NATS and log them."""
    
    print(f"Connecting to NATS at {nats_url}...")
    nc = await nats.connect(nats_url)
    print(f"Connected! Subscribing to: {subject}")
    print(f"Waiting for alerts (timeout: {timeout}s)...")
    print("=" * 70)
    
    async def message_handler(msg):
        try:
            alert = json.loads(msg.data.decode())
            received_alerts.append(alert)
            
            # Pretty print the alert
            risk_level = alert.get('risk_level', 'unknown').upper()
            risk_score = alert.get('risk_score', 0)
            tx_id = alert.get('transaction_id', 'unknown')
            
            # Color coding based on risk level
            if risk_level == 'CRITICAL':
                level_indicator = '🔴'
            elif risk_level == 'HIGH':
                level_indicator = '🟠'
            elif risk_level == 'MEDIUM':
                level_indicator = '🟡'
            else:
                level_indicator = '🟢'
            
            print(f"\n{level_indicator} ALERT #{len(received_alerts)}")
            print(f"   Transaction ID: {tx_id}")
            print(f"   Risk Score: {risk_score:.4f}")
            print(f"   Risk Level: {risk_level}")
            
            # Model scores
            model_scores = alert.get('model_scores', {})
            if model_scores:
                print(f"   Model Scores:")
                for model, score in sorted(model_scores.items(), key=lambda x: -x[1]):
                    print(f"      - {model}: {score:.4f}")
            
            # Triggered features
            triggered = alert.get('triggered_features', [])
            if triggered:
                print(f"   Triggered: {', '.join(triggered)}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    # Subscribe to alerts
    sub = await nc.subscribe(subject, cb=message_handler)
    
    # Wait for timeout
    try:
        await asyncio.sleep(timeout)
    except asyncio.CancelledError:
        pass
    
    await sub.unsubscribe()
    await nc.drain()
    
    # Print summary
    print("\n" + "=" * 70)
    print("ALERT SUMMARY")
    print("=" * 70)
    print(f"Total alerts received: {len(received_alerts)}")
    
    if received_alerts:
        # Risk level distribution
        risk_levels = {}
        for alert in received_alerts:
            level = alert.get('risk_level', 'unknown')
            risk_levels[level] = risk_levels.get(level, 0) + 1
        
        print("\nRisk Level Distribution:")
        for level, count in sorted(risk_levels.items()):
            print(f"  {level}: {count} ({count/len(received_alerts)*100:.1f}%)")
        
        # Average risk score
        avg_score = sum(a.get('risk_score', 0) for a in received_alerts) / len(received_alerts)
        print(f"\nAverage Risk Score: {avg_score:.4f}")
        
        # Save alerts to file
        output_file = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(received_alerts, f, indent=2)
        print(f"\nAlerts saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Receive alerts from NATS')
    parser.add_argument('--nats-url', default='nats://localhost:4222', help='NATS server URL')
    parser.add_argument('--subject', default='fraud.alerts', help='NATS subject for alerts')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    args = parser.parse_args()
    
    try:
        asyncio.run(receive_alerts(
            args.nats_url,
            args.subject,
            args.timeout
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == '__main__':
    main()


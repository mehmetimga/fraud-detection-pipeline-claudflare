#!/usr/bin/env python3
"""
Analyze fraud detection pipeline results.
Compare predicted alerts with actual defaults.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def load_alerts():
    """Load the most recent alerts file."""
    alert_files = sorted(Path('.').glob('alerts_*.json'), reverse=True)
    if not alert_files:
        print("No alert files found!")
        return []
    
    latest = alert_files[0]
    print(f"Loading alerts from: {latest}")
    with open(latest) as f:
        return json.load(f)

def load_test_data(limit=150):
    """Load test data with actual labels."""
    df = pd.read_parquet('training/data/test.parquet')
    return df.head(limit)

def analyze(alerts, test_df):
    """Analyze prediction quality."""
    
    print("\n" + "="*70)
    print("FRAUD DETECTION PIPELINE - PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Create predictions DataFrame
    alert_data = {str(a['transaction_id']): {
        'risk_score': a['risk_score'],
        'risk_level': a.get('risk_level', 'unknown'),
        'model_scores': a.get('model_scores', {})
    } for a in alerts}
    
    # Get IDs that were sent (all in test_df limit)
    test_df = test_df.copy()
    test_df['ID'] = test_df['ID'].astype(str)
    
    # Only analyze transactions that were sent (in test_df)
    sent_ids = set(test_df['ID'].tolist())
    alert_ids = set(alert_data.keys())
    
    # For analysis, we consider:
    # - Predicted default = in alerts (score > threshold, i.e., alert was generated)
    # - Predicted normal = not in alerts
    test_df['predicted_score'] = test_df['ID'].apply(
        lambda x: alert_data.get(x, {}).get('risk_score', 0.0)
    )
    test_df['has_alert'] = test_df['ID'].isin(alert_ids)
    
    # Confusion matrix based on whether alert was generated
    actual_defaults = test_df['is_default'] == 1
    has_alert = test_df['has_alert']
    
    tp = ((actual_defaults) & (has_alert)).sum()  # Default correctly flagged
    fp = ((~actual_defaults) & (has_alert)).sum()  # Normal incorrectly flagged
    tn = ((~actual_defaults) & (~has_alert)).sum()  # Normal correctly not flagged
    fn = ((actual_defaults) & (~has_alert)).sum()  # Default missed
    
    total_sent = len(test_df)
    actual_default_count = actual_defaults.sum()
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Total transactions sent: {total_sent}")
    print(f"   Actual defaults in test: {actual_default_count} ({actual_default_count/total_sent*100:.1f}%)")
    print(f"   Alerts generated: {len(alerts)} ({len(alerts)/total_sent*100:.1f}%)")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"                    Predicted (Alert Generated)")
    print(f"                    Alert      No Alert")
    print(f"   Actual Default     {tp:4d}        {fn:4d}")
    print(f"   Actual Normal      {fp:4d}        {tn:4d}")
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_sent
    
    print(f"\n📋 METRICS:")
    print(f"   Precision: {precision:.3f} (of alerts, how many were actual defaults)")
    print(f"   Recall:    {recall:.3f} (of actual defaults, how many were alerted)")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   Accuracy:  {accuracy:.3f}")
    
    # Risk level distribution
    print(f"\n🎯 RISK LEVEL DISTRIBUTION OF ALERTS:")
    risk_counts = {}
    for a in alerts:
        level = a.get('risk_level', 'unknown')
        risk_counts[level] = risk_counts.get(level, 0) + 1
    for level, count in sorted(risk_counts.items()):
        print(f"   {level.upper():12s}: {count:4d} ({count/len(alerts)*100:.1f}%)")
    
    # Score distribution for alerts
    alert_scores = [a['risk_score'] for a in alerts]
    print(f"\n📈 ALERT SCORE STATISTICS:")
    print(f"   Min:    {min(alert_scores):.3f}")
    print(f"   Max:    {max(alert_scores):.3f}")
    print(f"   Mean:   {sum(alert_scores)/len(alert_scores):.3f}")
    print(f"   Median: {sorted(alert_scores)[len(alert_scores)//2]:.3f}")
    
    # Model agreement analysis for true positives
    print(f"\n🤖 TOP CORRECTLY IDENTIFIED DEFAULTS:")
    # Get alerts that are true positives
    tp_alerts = []
    for a in alerts:
        tx_id = str(a['transaction_id'])
        row = test_df[test_df['ID'] == tx_id]
        if len(row) > 0 and row['is_default'].values[0] == 1:
            tp_alerts.append(a)
    
    tp_alerts_sorted = sorted(tp_alerts, key=lambda x: x['risk_score'], reverse=True)[:5]
    for i, a in enumerate(tp_alerts_sorted, 1):
        print(f"\n   #{i} Transaction {a['transaction_id']} ✓ CORRECTLY FLAGGED")
        print(f"      Risk Score: {a['risk_score']:.3f} [{a['risk_level'].upper()}]")
        scores = a.get('model_scores', {})
        for model, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"      {model:15s}: {score:.3f} {bar}")
    
    # False positives (incorrectly flagged as default)
    print(f"\n⚠️  FALSE POSITIVES (Normal flagged as default): {fp}")
    fp_alerts = []
    for a in alerts:
        tx_id = str(a['transaction_id'])
        row = test_df[test_df['ID'] == tx_id]
        if len(row) > 0 and row['is_default'].values[0] == 0:
            fp_alerts.append(a)
    
    fp_sorted = sorted(fp_alerts, key=lambda x: x['risk_score'], reverse=True)[:3]
    for a in fp_sorted:
        print(f"   - ID {a['transaction_id']}: score={a['risk_score']:.3f} [{a['risk_level']}]")
    
    # Missed defaults
    print(f"\n❌ FALSE NEGATIVES (Defaults not detected): {fn}")
    missed = test_df[(test_df['is_default'] == 1) & (~test_df['has_alert'])]
    if len(missed) > 0:
        # These had no alert - show some characteristics
        print(f"   Sample missed defaults:")
        for _, row in missed.head(5).iterrows():
            print(f"   - ID {row['ID']}: PAY_0={row['PAY_0']}, LIMIT_BAL={row['LIMIT_BAL']:.0f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def main():
    # Load data
    alerts = load_alerts()
    if not alerts:
        print("No alerts to analyze. Run the pipeline first.")
        return
    
    test_df = load_test_data(limit=150)
    
    # Analyze
    results = analyze(alerts, test_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

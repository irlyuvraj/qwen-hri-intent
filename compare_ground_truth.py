#!/usr/bin/env python3
"""
Ground Truth Comparison Tool

Compares system predictions against ground truth annotations.

Usage:
    python compare_ground_truth.py \
        predictions.jsonl \
        ground_truth.json \
        --output report.txt

Computes:
- Detection rate per intent class
- Mean advance time (how far ahead predictions occur)
- False positive rate
- Temporal accuracy
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys


class GroundTruthComparison:
    """Compare predictions to ground truth"""
    
    def __init__(self, predictions_file, ground_truth_file):
        self.predictions = self.load_predictions(predictions_file)
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.results = {
            'detections': defaultdict(list),
            'misses': defaultdict(list),
            'false_positives': [],
            'advance_times': defaultdict(list),
        }
    
    def load_predictions(self, filepath):
        """Load predictions from JSONL file"""
        predictions = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    pred = json.loads(line)
                    predictions.append(pred)
        
        print(f"✓ Loaded {len(predictions)} predictions from {filepath}")
        return predictions
    
    def load_ground_truth(self, filepath):
        """Load ground truth annotations from JSON file"""
        with open(filepath, 'r') as f:
            gt = json.load(f)
        
        annotations = gt.get('annotations', [])
        print(f"✓ Loaded {len(annotations)} ground truth events from {filepath}")
        return gt
    
    def get_prediction_time(self, pred):
        """Get timestamp of prediction"""
        # Try different possible field names
        if 'timestamp' in pred:
            timestamp = pred['timestamp']
            
            # Check if timestamp is Unix epoch (very large number)
            if timestamp > 1000000000:  # Year 2001+, definitely Unix epoch
                # Convert to relative time from first prediction
                if not hasattr(self, '_first_timestamp'):
                    self._first_timestamp = timestamp
                return timestamp - self._first_timestamp
            else:
                return timestamp
        elif 'sequence_id' in pred:
            # Assuming 2-second intervals
            return pred['sequence_id'] * 2.0
        else:
            return 0.0
    
    def get_intent(self, pred):
        """Get predicted intent"""
        if 'predicted_intent' in pred:
            return pred['predicted_intent']
        elif 'intent' in pred:
            return pred['intent']
        else:
            return 'unknown'
    
    def compare(self):
        """Main comparison logic"""
        
        annotations = self.ground_truth.get('annotations', [])
        
        # For each ground truth event, find matching predictions
        for event in annotations:
            event_id = event['event_id']
            gt_intent = event['intent']
            gt_start = event['start_time']
            gt_end = event['end_time']
            
            # Find predictions that match this event
            matches = []
            for pred in self.predictions:
                pred_time = self.get_prediction_time(pred)
                pred_intent = self.get_intent(pred)
                
                # Check if prediction is within or before event window
                # Prediction should occur between (start - 2s) and end
                # (system predicts up to 2 seconds ahead)
                if pred_time >= (gt_start - 2.5) and pred_time <= gt_end:
                    if pred_intent == gt_intent:
                        advance_time = gt_start - pred_time
                        matches.append({
                            'pred_time': pred_time,
                            'advance_time': advance_time,
                            'confidence': pred.get('confidence', 0.0)
                        })
            
            if matches:
                # Event was detected
                self.results['detections'][gt_intent].append({
                    'event_id': event_id,
                    'matches': matches,
                    'gt_start': gt_start,
                    'gt_end': gt_end
                })
                
                # Record advance times
                for match in matches:
                    if match['advance_time'] > 0:  # Only count if prediction was before event
                        self.results['advance_times'][gt_intent].append(match['advance_time'])
            else:
                # Event was missed
                self.results['misses'][gt_intent].append({
                    'event_id': event_id,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'description': event.get('description', '')
                })
        
        # Find false positives (predictions that don't match any ground truth)
        for pred in self.predictions:
            pred_time = self.get_prediction_time(pred)
            pred_intent = self.get_intent(pred)
            
            # Check if this prediction matches any ground truth event
            matched = False
            for event in annotations:
                gt_intent = event['intent']
                gt_start = event['start_time']
                gt_end = event['end_time']
                
                if pred_intent == gt_intent:
                    if pred_time >= (gt_start - 2.5) and pred_time <= gt_end:
                        matched = True
                        break
            
            if not matched:
                self.results['false_positives'].append({
                    'time': pred_time,
                    'intent': pred_intent,
                    'confidence': pred.get('confidence', 0.0)
                })
    
    def print_report(self, output_file=None):
        """Print comparison report"""
        
        lines = []
        
        def add_line(text=""):
            lines.append(text)
            print(text)
        
        add_line("=" * 70)
        add_line("GROUND TRUTH COMPARISON REPORT")
        add_line("=" * 70)
        add_line()
        
        # Summary statistics
        annotations = self.ground_truth.get('annotations', [])
        total_events = len(annotations)
        total_detected = sum(len(v) for v in self.results['detections'].values())
        total_missed = sum(len(v) for v in self.results['misses'].values())
        total_fps = len(self.results['false_positives'])
        
        add_line(f"Total Ground Truth Events: {total_events}")
        add_line(f"Total Detected: {total_detected}")
        add_line(f"Total Missed: {total_missed}")
        add_line(f"Total False Positives: {total_fps}")
        add_line(f"Overall Detection Rate: {total_detected/total_events*100:.1f}%")
        add_line()
        
        # Per-intent breakdown
        add_line("=" * 70)
        add_line("DETECTION RATE BY INTENT CLASS")
        add_line("=" * 70)
        add_line()
        
        intent_counts = defaultdict(int)
        for event in annotations:
            intent_counts[event['intent']] += 1
        
        add_line(f"{'Intent':<15} {'GT Events':<12} {'Detected':<12} {'Detection Rate':<15} {'Avg Advance'}")
        add_line("-" * 70)
        
        for intent in sorted(intent_counts.keys()):
            gt_count = intent_counts[intent]
            detected_count = len(self.results['detections'][intent])
            detection_rate = detected_count / gt_count * 100 if gt_count > 0 else 0
            
            advance_times = self.results['advance_times'][intent]
            avg_advance = sum(advance_times) / len(advance_times) if advance_times else 0
            
            add_line(f"{intent:<15} {gt_count:<12} {detected_count:<12} {detection_rate:>6.1f}%         {avg_advance:>6.2f}s")
        
        add_line()
        
        # Detailed missed events
        if any(self.results['misses'].values()):
            add_line("=" * 70)
            add_line("MISSED EVENTS (Not Detected)")
            add_line("=" * 70)
            add_line()
            
            for intent, misses in self.results['misses'].items():
                if misses:
                    add_line(f"{intent.upper()}:")
                    for miss in misses:
                        add_line(f"  Event {miss['event_id']}: t={miss['gt_start']:.1f}-{miss['gt_end']:.1f}s")
                        add_line(f"    {miss['description']}")
                    add_line()
        
        # False positives
        if self.results['false_positives']:
            add_line("=" * 70)
            add_line(f"FALSE POSITIVES ({len(self.results['false_positives'])} total)")
            add_line("=" * 70)
            add_line()
            
            fp_by_intent = defaultdict(list)
            for fp in self.results['false_positives']:
                fp_by_intent[fp['intent']].append(fp)
            
            for intent, fps in sorted(fp_by_intent.items()):
                add_line(f"{intent}: {len(fps)} false positives")
                for fp in fps[:5]:  # Show first 5
                    add_line(f"  t={fp['time']:.1f}s (conf={fp['confidence']:.2f})")
                if len(fps) > 5:
                    add_line(f"  ... and {len(fps)-5} more")
                add_line()
        
        # Advance time analysis
        if any(self.results['advance_times'].values()):
            add_line("=" * 70)
            add_line("ADVANCE TIME ANALYSIS")
            add_line("=" * 70)
            add_line()
            add_line("How far in advance did the system predict each intent?")
            add_line("(Positive = prediction before event start, ideal range: 1-2 seconds)")
            add_line()
            
            for intent, times in sorted(self.results['advance_times'].items()):
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    add_line(f"{intent:<15}: avg={avg_time:>5.2f}s, min={min_time:>5.2f}s, max={max_time:>5.2f}s ({len(times)} detections)")
            add_line()
        
        add_line("=" * 70)
        add_line("INTERPRETATION GUIDE")
        add_line("=" * 70)
        add_line()
        add_line("Detection Rate:")
        add_line("  >80% = Excellent")
        add_line("  60-80% = Good")
        add_line("  40-60% = Fair")
        add_line("  <40% = Poor")
        add_line()
        add_line("Advance Time:")
        add_line("  1-2s = Ideal (predicts actions with useful lead time)")
        add_line("  0-1s = Acceptable (some prediction capability)")
        add_line("  <0s = Late (prediction after event already started)")
        add_line()
        add_line("False Positive Rate:")
        add_line("  <10% = Excellent")
        add_line("  10-20% = Acceptable")
        add_line("  >20% = Concerning (system hallucinating too much)")
        add_line()
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(lines))
            print(f"\n✓ Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare predictions to ground truth")
    parser.add_argument('predictions', help='Predictions JSONL file')
    parser.add_argument('ground_truth', help='Ground truth JSON file')
    parser.add_argument('--output', '-o', help='Output report file')
    
    args = parser.parse_args()
    
    if not Path(args.predictions).exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    if not Path(args.ground_truth).exists():
        print(f"ERROR: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    
    # Run comparison
    comparison = GroundTruthComparison(args.predictions, args.ground_truth)
    comparison.compare()
    comparison.print_report(args.output)


if __name__ == "__main__":
    main()

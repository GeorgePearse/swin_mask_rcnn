"""Quick script to monitor predictions during training."""
import time
import subprocess
import re

def monitor_predictions(duration_seconds=60):
    """Monitor predictions for specified duration."""
    print(f"Monitoring predictions for {duration_seconds} seconds...")
    
    # Start training process
    process = subprocess.Popen([
        'python', 'scripts/train.py', 
        '--config', 'scripts/config/train_with_fixed_biases.yaml'
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    start_time = time.time()
    validation_count = 0
    prediction_counts = []
    
    try:
        for line in iter(process.stdout.readline, ''):
            if time.time() - start_time > duration_seconds:
                break
                
            # Look for validation predictions
            if "Saving" in line and "predictions to" in line:
                match = re.search(r'Saving (\d+) predictions', line)
                if match:
                    pred_count = int(match.group(1))
                    prediction_counts.append(pred_count)
                    validation_count += 1
                    print(f"Validation {validation_count}: {pred_count} predictions")
            
            # Look for metrics
            if "mAP:" in line or "Detection mAP50:" in line:
                print(line.strip())
                
    finally:
        process.terminate()
        process.wait()
    
    print(f"\nSummary after {duration_seconds} seconds:")
    print(f"Validation runs: {validation_count}")
    if prediction_counts:
        print(f"Prediction counts: {prediction_counts}")
        print(f"Average predictions: {sum(prediction_counts)/len(prediction_counts):.0f}")

if __name__ == "__main__":
    monitor_predictions(duration_seconds=120)  # Monitor for 2 minutes
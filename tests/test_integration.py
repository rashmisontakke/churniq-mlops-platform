import os
import subprocess
import pandas as pd

def test_full_pipeline():
    # Remove old artifacts if they exist
    for f in ['models/model.pkl', 'models/model_columns.pkl', 'models/predictions.csv']:
        if os.path.exists(f):
            os.remove(f)
    # Run training
    result = subprocess.run(['python', 'src/train.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"train.py failed: {result.stderr}"
    assert os.path.exists('models/model.pkl'), 'model.pkl not created.'
    # Run batch prediction
    result = subprocess.run(['python', 'src/batch.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"batch.py failed: {result.stderr}"
    assert os.path.exists('models/predictions.csv'), 'predictions.csv not created.'
    # Run monitor
    result = subprocess.run(['python', 'src/monitor.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"monitor.py failed: {result.stderr}"
    # Check predictions.csv content
    preds = pd.read_csv('models/predictions.csv')
    assert 'prediction' in preds.columns, 'prediction column missing in predictions.csv'
    assert len(preds) > 0, 'No predictions made.' 
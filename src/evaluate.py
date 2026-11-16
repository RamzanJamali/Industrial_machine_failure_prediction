import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, output_dir='industrial-machine-failure-prediction/reports'):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("--- Evaluation Report ---")
    print(report)
    
    # Save metrics to reports folder
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print(f"Reports saved to {output_dir}/")
    return report, cm

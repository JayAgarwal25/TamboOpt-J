#!/usr/bin/env python3
"""
run_notebook.py - Execute Jupyter notebooks with papermill
Install: pip install papermill
"""

import papermill as pm
import sys
from pathlib import Path
from datetime import datetime

def run_notebook(input_path, output_dir="./outputs_notebooks", parameters=None):
    """
    Execute a Jupyter notebook and save results
    
    Args:
        input_path: Path to input notebook
        output_dir: Directory to save output
        parameters: Dict of parameters to inject
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{input_path.stem}_output_{timestamp}.ipynb"
    
    print(f"Executing: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        pm.execute_notebook(
            str(input_path),
            str(output_path),
            parameters=parameters or {},
            kernel_name='python3',
            progress_bar=True,
            log_output=True,
            timeout=-1  # No timeout
        )
        print(f"✓ Success! Results saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb> [param1=value1] [param2=value2]")
        sys.exit(1)
    
    notebook = sys.argv[1]
    
    # Parse parameters from command line
    params = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    run_notebook(notebook, parameters=params)

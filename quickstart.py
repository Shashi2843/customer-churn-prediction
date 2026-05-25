"""
Quick start script for Customer Churn Prediction System
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command"""
    print(f"\n{'='*80}")
    print(f">> {description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    """Setup and run quick start"""
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║      CUSTOMER CHURN PREDICTION SYSTEM - QUICK START SETUP                  ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Create virtual environment
    print("\n[Step 1] Creating Python virtual environment...")
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists")
    else:
        if not run_command("python -m venv venv", "Creating virtual environment"):
            print("✗ Failed to create virtual environment")
            return
        print("✓ Virtual environment created")
    
    # Step 2: Activate and install dependencies
    activate_cmd = "venv\\Scripts\\activate" if sys.platform == "win32" else "source venv/bin/activate"
    
    print("\n[Step 2] Installing dependencies...")
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip install -r requirements.txt"
    else:
        pip_cmd = "venv/bin/pip install -r requirements.txt"
    
    if not run_command(pip_cmd, "Installing Python packages"):
        print("✗ Failed to install dependencies")
        return
    print("✓ Dependencies installed")
    
    # Step 3: Create .env file
    print("\n[Step 3] Setting up environment variables...")
    env_file = Path(".env")
    if not env_file.exists():
        with open(".env", "w") as f:
            f.write("MLFLOW_TRACKING_URI=sqlite:///mlruns.db\n")
            f.write("LOG_LEVEL=INFO\n")
        print("✓ .env file created")
    else:
        print("✓ .env file already exists")
    
    # Step 4: Download dataset
    print("\n[Step 4] Downloading dataset...")
    if sys.platform == "win32":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    run_command(
        f"{python_cmd} -c \"from src.data_loader import download_dataset; download_dataset()\"",
        "Downloading IBM Telco Customer Churn dataset"
    )
    print("✓ Dataset downloaded")
    
    # Step 5: Run EDA
    print("\n[Step 5] Running Exploratory Data Analysis...")
    run_command(
        f"{python_cmd} notebooks/01_eda.py",
        "Exploratory Data Analysis"
    )
    
    # Step 6: Run training pipeline
    print("\n[Step 6] Running training pipeline...")
    run_command(
        f"{python_cmd} train_pipeline.py",
        "Training ML models"
    )
    
    # Step 7: Run evaluation
    print("\n[Step 7] Running model evaluation...")
    run_command(
        f"{python_cmd} notebooks/04_evaluation_shap.py",
        "Model Evaluation & SHAP Analysis"
    )
    
    print(f"\n{'='*80}")
    print("✓ SETUP COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("""
    Next steps:
    
    1. Start the API server:
       uvicorn api.main:app --reload
    
    2. In another terminal, start the Streamlit dashboard:
       streamlit run dashboard/app.py
    
    3. Access the applications:
       - API Documentation: http://localhost:8000/docs
       - Dashboard: http://localhost:8501
    
    4. Run tests:
       pytest tests/
    
    For Docker deployment:
       docker-compose up
    """)

if __name__ == "__main__":
    main()

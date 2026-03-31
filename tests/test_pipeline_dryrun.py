import subprocess
import sys
from pathlib import Path

def test_pipeline_dryrun():
    """
    Run the pipeline with quick_test.yaml.
    This simulates an end-to-end run on synthetic data if the real dataset isn't fully available
    or just uses the quick limits defined in the yaml.
    """
    root_dir = Path(__file__).parent.parent
    script_path = root_dir / "pipelines" / "run_full_pipeline.py"
    config_path = root_dir / "configs" / "quick_test.yaml"
    
    # Run pipeline as a subprocess to verify the entry point
    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        cwd=str(root_dir),
        capture_output=True,
        text=True
    )
    
    # Check if execution was successful
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        assert False, f"Pipeline dry-run failed with return code {result.returncode}"
    
    # Specifically ensure the warnings we fixed aren't failing things
    assert "Traceback" not in result.stderr
    print("Dry run passed successfully!")

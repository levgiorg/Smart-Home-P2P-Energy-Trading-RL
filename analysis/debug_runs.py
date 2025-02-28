import json
from pathlib import Path
from typing import Dict, List
import re


class MechanismAnalyzer:
    def __init__(self, base_dir: str = "ml-outputs2"):
        self.base_dir = Path(base_dir)
        self.runs = self._find_run_directories()

    def _find_run_directories(self) -> List[str]:
        """Find all run directories that match the pattern run_X"""
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Directory '{self.base_dir}' not found")
        
        run_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and re.match(r"run_\d+", item.name):
                run_dirs.append(str(item))
        return sorted(run_dirs, key=lambda x: int(re.search(r"run_(\d+)", x).group(1)))

    def _get_mechanism_type(self, run_path: str) -> str:
        """Read mechanism type from hyperparameters.json"""
        try:
            hyperparams_path = Path(run_path) / 'hyperparameters.json'
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                
            # Check if anti_cartel mechanism is defined
            if 'anti_cartel' in hyperparams:
                mech_type = hyperparams['anti_cartel'].get('mechanism_type')
                # Handle the three possible cases
                if mech_type == "detection":
                    return "detection"
                elif mech_type == "ceiling":
                    return "ceiling"
                else:  # This will catch both null and any other unexpected values
                    return "null"
            return "null"  # Default to null if no anti_cartel section
        except Exception as e:
            print(f"Error reading mechanism type for run {run_path}: {e}")
            return 'unknown'

    def _group_runs_by_mechanism(self) -> Dict[str, List[str]]:
        """Group runs by their anti-cartel mechanism type"""
        groups = {'detection': [], 'ceiling': [], 'null': [], 'unknown': []}
        
        for run_path in self.runs:
            mech_type = self._get_mechanism_type(run_path)
            if mech_type in groups:
                groups[mech_type].append(run_path)
            else:
                print(f"Warning: Unknown mechanism type {mech_type} in {run_path}")
                groups['unknown'].append(run_path)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def analyze_and_print_results(self):
        """Analyze the runs and print results in a readable format"""
        grouped_runs = self._group_runs_by_mechanism()
        
        print("\nMechanism Type Analysis Results:")
        print("=" * 40)
        
        for mech_type, runs in grouped_runs.items():
            print(f"\n{mech_type.upper()} Mechanism:")
            print("-" * 20)
            for run in runs:
                run_number = re.search(r"run_(\d+)", run).group(1)
                print(f"Run {run_number}")
            print(f"Total {mech_type} runs: {len(runs)}")
        
        print("\nSummary:")
        print("-" * 20)
        for mech_type, runs in grouped_runs.items():
            print(f"{mech_type}: {len(runs)} runs")


if __name__ == "__main__":
    try:
        analyzer = MechanismAnalyzer()
        analyzer.analyze_and_print_results()
    except Exception as e:
        print(f"Error: {e}")
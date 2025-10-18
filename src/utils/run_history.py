"""
Run History Manager
Tracks and persists all ML pipeline executions
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class RunHistory:
    """Manages pipeline run history"""
    
    def __init__(self, history_dir: str = "history"):
        """
        Initialize run history manager
        
        Args:
            history_dir: Directory to store run history
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        self.index_file = self.history_dir / "run_index.json"
        self.runs = self._load_index()
    
    def _load_index(self) -> List[Dict[str, Any]]:
        """Load run index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_index(self):
        """Save run index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def save_run(
        self,
        results: Dict[str, Any],
        dataset_name: str,
        task_type: str,
        domain: str,
        target_column: str,
        dataset_shape: tuple,
        run_name: Optional[str] = None
    ) -> str:
        """
        Save a pipeline run
        
        Args:
            results: Full pipeline results
            dataset_name: Name of the dataset
            task_type: Classification or regression
            domain: Healthcare, finance, or general
            target_column: Target variable name
            dataset_shape: (rows, columns) tuple
            run_name: Optional custom name for the run
            
        Returns:
            run_id: Unique identifier for this run
        """
        timestamp = datetime.now()
        run_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Extract key metrics
        final_results = results.get('final_results', {})
        metrics = final_results.get('metrics', {})
        
        # Create run metadata
        run_metadata = {
            'run_id': run_id,
            'run_name': run_name or f"Run {timestamp.strftime('%Y-%m-%d %H:%M')}",
            'timestamp': timestamp.isoformat(),
            'dataset_name': dataset_name,
            'dataset_shape': list(dataset_shape),
            'task_type': task_type,
            'domain': domain,
            'target_column': target_column,
            'model_name': final_results.get('model_name', 'Unknown'),
            'deployment_ready': final_results.get('deployment_ready', False),
            'metrics': self._extract_key_metrics(metrics, task_type),
            'iterations': len(results.get('all_iterations', {})),
        }
        
        # Save full results to separate file
        run_file = self.history_dir / f"run_{run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Add to index
        self.runs.insert(0, run_metadata)  # Most recent first
        self._save_index()
        
        return run_id
    
    def _extract_key_metrics(self, metrics: Dict, task_type: str) -> Dict[str, float]:
        """Extract key metrics based on task type"""
        if task_type == "classification":
            return {
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
            }
        else:  # regression
            return {
                'r2_score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0),
            }
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all run metadata (most recent first)"""
        return self.runs
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full results for a specific run
        
        Args:
            run_id: Run identifier
            
        Returns:
            Full pipeline results or None if not found
        """
        run_file = self.history_dir / f"run_{run_id}.json"
        if run_file.exists():
            with open(run_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific run"""
        for run in self.runs:
            if run['run_id'] == run_id:
                return run
        return None
    
    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run from history
        
        Args:
            run_id: Run identifier
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from index
        self.runs = [r for r in self.runs if r['run_id'] != run_id]
        self._save_index()
        
        # Delete run file
        run_file = self.history_dir / f"run_{run_id}.json"
        if run_file.exists():
            run_file.unlink()
            return True
        return False
    
    def get_comparison_data(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Get comparison data for multiple runs
        
        Args:
            run_ids: List of run identifiers to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for run_id in run_ids:
            metadata = self.get_run_metadata(run_id)
            if metadata:
                row = {
                    'Run Name': metadata['run_name'],
                    'Dataset': metadata['dataset_name'],
                    'Model': metadata['model_name'],
                    'Task': metadata['task_type'].capitalize(),
                    'Domain': metadata['domain'].capitalize(),
                    'Deployed': '✅' if metadata['deployment_ready'] else '❌',
                    'Timestamp': metadata['timestamp'][:19],
                }
                # Add metrics
                row.update(metadata['metrics'])
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        if not self.runs:
            return {
                'total_runs': 0,
                'successful_deployments': 0,
                'most_used_model': 'N/A',
                'best_accuracy': 0,
            }
        
        total_runs = len(self.runs)
        successful_deployments = sum(1 for r in self.runs if r['deployment_ready'])
        
        # Most used model
        model_counts = {}
        for run in self.runs:
            model = run['model_name']
            model_counts[model] = model_counts.get(model, 0) + 1
        most_used_model = max(model_counts, key=model_counts.get) if model_counts else 'N/A'
        
        # Best metrics (classification)
        accuracies = [
            r['metrics'].get('accuracy', 0) 
            for r in self.runs 
            if 'accuracy' in r['metrics']
        ]
        best_accuracy = max(accuracies) if accuracies else 0
        
        return {
            'total_runs': total_runs,
            'successful_deployments': successful_deployments,
            'most_used_model': most_used_model,
            'best_accuracy': best_accuracy,
        }
    
    def export_run(self, run_id: str, export_path: str) -> bool:
        """
        Export a run to a specific location
        
        Args:
            run_id: Run identifier
            export_path: Path to export the run
            
        Returns:
            True if exported successfully
        """
        results = self.get_run(run_id)
        if results:
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            return True
        return False


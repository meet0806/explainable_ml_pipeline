"""
Main Entry Point for Explainable ML Agentic Pipeline
Healthcare and Finance Domain Application
"""

import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path

from src.orchestrator import Orchestrator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    # Create handlers with UTF-8 encoding
    import sys
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('logs/pipeline.log', encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler, file_handler]
    )
    
    # Force UTF-8 encoding for stdout if on Windows
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from file"""
    file_ext = Path(data_path).suffix.lower()
    
    if file_ext == '.csv':
        return pd.read_csv(data_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(data_path)
    elif file_ext == '.parquet':
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Explainable ML Pipeline with Agentic AI"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input dataset (CSV, Excel, or Parquet)'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Name of target column'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression'],
        default='classification',
        help='Type of ML task'
    )
    parser.add_argument(
        '--domain',
        type=str,
        choices=['healthcare', 'finance', 'general'],
        default='general',
        help='Domain context for the problem'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save final model to disk'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("Explainable ML Pipeline - Starting")
    logger.info("="*80)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Load data
        logger.info(f"Loading data from: {args.data}")
        data = load_data(args.data)
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Target column: {args.target}")
        logger.info(f"Task type: {args.task}")
        logger.info(f"Domain: {args.domain}")
        
        # Verify target column exists
        if args.target not in data.columns:
            raise ValueError(f"Target column '{args.target}' not found in dataset")
        
        # Initialize orchestrator
        logger.info("\nInitializing Orchestrator and Agents...")
        orchestrator = Orchestrator(config)
        
        # Run pipeline
        logger.info("\nStarting Pipeline Execution...")
        results = orchestrator.run_pipeline(
            data=data,
            target_column=args.target,
            task_type=args.task,
            domain=args.domain
        )
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("PIPELINE RESULTS")
        logger.info("="*80)
        
        final_results = results.get("final_results", {})
        logger.info(f"\nModel Approved: {final_results.get('model_approved')}")
        logger.info(f"Deployment Ready: {final_results.get('deployment_ready')}")
        logger.info(f"Best Model: {final_results.get('best_model')}")
        logger.info(f"Performance Score: {final_results.get('performance_score'):.4f}")
        
        # Display metrics
        logger.info("\nPerformance Metrics:")
        metrics = final_results.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Display recommendations
        logger.info("\nRecommendations:")
        for rec in results.get("recommendations", []):
            logger.info(f"  {rec}")
        
        # Save model if requested
        if args.save_model:
            logger.info("\nSaving final model...")
            orchestrator.save_final_model()
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline Execution Completed Successfully")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


"""Benchmarking and performance testing module."""

import torch
import numpy as np
import time
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..core.logger import setup_logger
from ..core.exceptions import ProcessingError
from ..model3d_integration.text_to_shape import TextToShape
from ..model3d_integration.image_to_shape import ImageToShape
from ..evaluation.metrics import ModelEvaluator

logger = setup_logger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float]
    gpu_utilization: Optional[float]

@dataclass
class TimingMetrics:
    """Timing metrics."""
    total_time: float
    preprocessing_time: float
    inference_time: float
    postprocessing_time: float

@dataclass
class QualityMetrics:
    """Quality metrics."""
    chamfer_distance: float
    iou_score: float
    f_score: float
    surface_metrics: Dict[str, float]

class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.has_gpu = torch.cuda.is_available()
    
    def measure_resource_usage(self) -> ResourceMetrics:
        """Measure current resource usage."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # GPU usage
            gpu_memory_percent = None
            gpu_utilization = None
            
            if self.has_gpu:
                gpu = GPUtil.getGPUs()[0]
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_utilization = gpu.load * 100
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_percent=gpu_memory_percent,
                gpu_utilization=gpu_utilization
            )
            
        except Exception as e:
            logger.error(f"Resource measurement failed: {str(e)}")
            return ResourceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_memory_percent=None,
                gpu_utilization=None
            )
    
    def measure_timing(
        self,
        func: callable,
        *args,
        **kwargs
    ) -> TimingMetrics:
        """Measure execution timing."""
        try:
            # Total time
            start_total = time.time()
            
            # Preprocessing time
            start_prep = time.time()
            if hasattr(func, "preprocess"):
                func.preprocess(*args, **kwargs)
            prep_time = time.time() - start_prep
            
            # Inference time
            start_inf = time.time()
            result = func(*args, **kwargs)
            inf_time = time.time() - start_inf
            
            # Postprocessing time
            start_post = time.time()
            if hasattr(func, "postprocess"):
                func.postprocess(result)
            post_time = time.time() - start_post
            
            total_time = time.time() - start_total
            
            return TimingMetrics(
                total_time=total_time,
                preprocessing_time=prep_time,
                inference_time=inf_time,
                postprocessing_time=post_time
            )
            
        except Exception as e:
            logger.error(f"Timing measurement failed: {str(e)}")
            return TimingMetrics(
                total_time=0.0,
                preprocessing_time=0.0,
                inference_time=0.0,
                postprocessing_time=0.0
            )
    
    def measure_quality(
        self,
        prediction: Dict[str, Any],
        target: Dict[str, Any]
    ) -> QualityMetrics:
        """Measure output quality metrics."""
        try:
            metrics = self.evaluator.evaluate_single(prediction, target)
            
            return QualityMetrics(
                chamfer_distance=metrics.get("chamfer_distance", 0.0),
                iou_score=metrics.get("iou", 0.0),
                f_score=metrics.get("f_score", 0.0),
                surface_metrics=metrics.get("surface_metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Quality measurement failed: {str(e)}")
            return QualityMetrics(
                chamfer_distance=0.0,
                iou_score=0.0,
                f_score=0.0,
                surface_metrics={}
            )

class ModelBenchmark:
    """Model benchmarking utility."""
    
    def __init__(
        self,
        model_type: str = "text_to_shape",
        device: Optional[torch.device] = None
    ):
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = PerformanceMonitor()
        
        # Initialize model
        if model_type == "text_to_shape":
            self.model = TextToShape(device=self.device)
        elif model_type == "image_to_shape":
            self.model = ImageToShape(device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def run_benchmark(
        self,
        test_cases: List[Dict[str, Any]],
        num_runs: int = 5,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run benchmark tests."""
        try:
            results = {
                "model_type": self.model_type,
                "device": str(self.device),
                "num_runs": num_runs,
                "test_cases": []
            }
            
            for case_idx, test_case in enumerate(test_cases):
                logger.info(f"Running test case {case_idx + 1}/{len(test_cases)}")
                
                case_results = {
                    "case_id": case_idx,
                    "input": test_case,
                    "runs": []
                }
                
                for run in tqdm(range(num_runs), desc=f"Test case {case_idx + 1}"):
                    # Measure performance
                    resource_metrics = self.monitor.measure_resource_usage()
                    timing_metrics = self.monitor.measure_timing(
                        self.model.predict,
                        test_case["input"]
                    )
                    
                    if "target" in test_case:
                        quality_metrics = self.monitor.measure_quality(
                            {"mesh": self.model.predict(test_case["input"])},
                            {"mesh": test_case["target"]}
                        )
                    else:
                        quality_metrics = None
                    
                    # Record results
                    run_results = {
                        "run_id": run,
                        "resources": {
                            "cpu_percent": resource_metrics.cpu_percent,
                            "memory_percent": resource_metrics.memory_percent,
                            "gpu_memory_percent": resource_metrics.gpu_memory_percent,
                            "gpu_utilization": resource_metrics.gpu_utilization
                        },
                        "timing": {
                            "total_time": timing_metrics.total_time,
                            "preprocessing_time": timing_metrics.preprocessing_time,
                            "inference_time": timing_metrics.inference_time,
                            "postprocessing_time": timing_metrics.postprocessing_time
                        }
                    }
                    
                    if quality_metrics:
                        run_results["quality"] = {
                            "chamfer_distance": quality_metrics.chamfer_distance,
                            "iou_score": quality_metrics.iou_score,
                            "f_score": quality_metrics.f_score,
                            "surface_metrics": quality_metrics.surface_metrics
                        }
                    
                    case_results["runs"].append(run_results)
                
                # Compute statistics for the test case
                case_results["statistics"] = self._compute_statistics(case_results["runs"])
                results["test_cases"].append(case_results)
            
            # Save results if output directory is provided
            if output_dir:
                self._save_results(results, output_dir)
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise ProcessingError(f"Benchmark failed: {str(e)}")
    
    def _compute_statistics(
        self,
        runs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics from multiple runs."""
        stats = {
            "resources": {},
            "timing": {}
        }
        
        # Resource statistics
        for metric in ["cpu_percent", "memory_percent", "gpu_memory_percent", "gpu_utilization"]:
            values = [run["resources"][metric] for run in runs if run["resources"][metric] is not None]
            if values:
                stats["resources"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Timing statistics
        for metric in ["total_time", "preprocessing_time", "inference_time", "postprocessing_time"]:
            values = [run["timing"][metric] for run in runs]
            stats["timing"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        # Quality statistics if available
        if "quality" in runs[0]:
            stats["quality"] = {}
            for metric in ["chamfer_distance", "iou_score", "f_score"]:
                values = [run["quality"][metric] for run in runs]
                stats["quality"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return stats
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save benchmark results."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw results
            with open(output_dir / "benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Generate plots
            self._generate_plots(results, output_dir)
            
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def _generate_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate visualization plots."""
        try:
            # Timing distribution plot
            plt.figure(figsize=(10, 6))
            for case in results["test_cases"]:
                times = [run["timing"]["total_time"] for run in case["runs"]]
                plt.hist(times, alpha=0.5, label=f"Case {case['case_id']}")
            plt.xlabel("Total Time (s)")
            plt.ylabel("Frequency")
            plt.title("Execution Time Distribution")
            plt.legend()
            plt.savefig(output_dir / "timing_distribution.png")
            plt.close()
            
            # Resource usage plot
            plt.figure(figsize=(12, 6))
            case_ids = [case["case_id"] for case in results["test_cases"]]
            cpu_usage = [case["statistics"]["resources"]["cpu_percent"]["mean"]
                        for case in results["test_cases"]]
            memory_usage = [case["statistics"]["resources"]["memory_percent"]["mean"]
                          for case in results["test_cases"]]
            
            x = np.arange(len(case_ids))
            width = 0.35
            
            plt.bar(x - width/2, cpu_usage, width, label="CPU Usage (%)")
            plt.bar(x + width/2, memory_usage, width, label="Memory Usage (%)")
            
            plt.xlabel("Test Case")
            plt.ylabel("Usage (%)")
            plt.title("Resource Usage by Test Case")
            plt.xticks(x, [f"Case {id}" for id in case_ids])
            plt.legend()
            plt.savefig(output_dir / "resource_usage.png")
            plt.close()
            
            # Quality metrics plot if available
            if "quality" in results["test_cases"][0]["statistics"]:
                plt.figure(figsize=(10, 6))
                metrics = ["chamfer_distance", "iou_score", "f_score"]
                for metric in metrics:
                    values = [case["statistics"]["quality"][metric]["mean"]
                            for case in results["test_cases"]]
                    plt.plot(case_ids, values, marker="o", label=metric)
                
                plt.xlabel("Test Case")
                plt.ylabel("Score")
                plt.title("Quality Metrics by Test Case")
                plt.legend()
                plt.savefig(output_dir / "quality_metrics.png")
                plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")

def run_comprehensive_benchmark(
    test_suite: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run comprehensive benchmark suite."""
    try:
        results = {}
        
        for model_type, test_cases in test_suite.items():
            logger.info(f"Benchmarking {model_type} model...")
            
            benchmark = ModelBenchmark(model_type=model_type)
            model_results = benchmark.run_benchmark(
                test_cases,
                output_dir=output_dir / model_type if output_dir else None
            )
            
            results[model_type] = model_results
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive benchmark failed: {str(e)}")
        raise ProcessingError(f"Comprehensive benchmark failed: {str(e)}") 
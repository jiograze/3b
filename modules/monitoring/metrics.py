"""Monitoring and alerting system for Ötüken3D."""

import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import psutil
import GPUtil
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import json
from pathlib import Path
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..core.logger import setup_logger
from ..core.exceptions import MonitoringError

logger = setup_logger(__name__)

# Prometheus metrics
METRICS = {
    # Resource metrics
    "cpu_usage": Gauge("cpu_usage_percent", "CPU usage percentage"),
    "memory_usage": Gauge("memory_usage_percent", "Memory usage percentage"),
    "gpu_usage": Gauge("gpu_usage_percent", "GPU usage percentage"),
    "gpu_memory": Gauge("gpu_memory_percent", "GPU memory usage percentage"),
    
    # Model metrics
    "inference_time": Histogram(
        "model_inference_time_seconds",
        "Model inference time in seconds",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ),
    "model_accuracy": Gauge("model_accuracy", "Model accuracy score"),
    "model_loss": Gauge("model_loss", "Model training loss"),
    
    # API metrics
    "api_requests": Counter("api_requests_total", "Total API requests"),
    "api_errors": Counter("api_errors_total", "Total API errors"),
    "api_latency": Summary("api_latency_seconds", "API request latency"),
    
    # System metrics
    "disk_usage": Gauge("disk_usage_percent", "Disk usage percentage"),
    "network_io": Gauge("network_io_bytes", "Network I/O bytes"),
}

@dataclass
class Alert:
    """Alert configuration."""
    name: str
    condition: str
    threshold: float
    message: str
    severity: str
    cooldown: int  # seconds

class MetricsCollector:
    """System metrics collection utility."""
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics."""
        try:
            metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "network_io": sum(psutil.net_io_counters()[:2])
            }
            
            if self.has_gpu:
                gpu = GPUtil.getGPUs()[0]
                metrics.update({
                    "gpu_usage": gpu.load * 100,
                    "gpu_memory": gpu.memoryUtil * 100
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}

class AlertManager:
    """Alert management system."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.alerts: Dict[str, Alert] = {}
        self.last_triggered: Dict[str, float] = {}
        self.email_config = email_config
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Path):
        """Load alert configuration from file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            for alert_config in config["alerts"]:
                self.add_alert(Alert(**alert_config))
                
        except Exception as e:
            raise MonitoringError(f"Failed to load alert config: {str(e)}")
    
    def add_alert(self, alert: Alert):
        """Add new alert configuration."""
        self.alerts[alert.name] = alert
        self.last_triggered[alert.name] = 0
    
    def check_condition(
        self,
        alert: Alert,
        value: float
    ) -> bool:
        """Check if alert condition is met."""
        try:
            condition = alert.condition
            threshold = alert.threshold
            
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            else:
                raise ValueError(f"Invalid condition: {condition}")
                
        except Exception as e:
            logger.error(f"Failed to check condition: {str(e)}")
            return False
    
    def should_trigger(
        self,
        alert: Alert,
        current_time: float
    ) -> bool:
        """Check if alert should be triggered based on cooldown."""
        last_time = self.last_triggered.get(alert.name, 0)
        return (current_time - last_time) >= alert.cooldown
    
    def send_email_alert(
        self,
        alert: Alert,
        value: float
    ):
        """Send email alert."""
        try:
            if not self.email_config:
                return
            
            msg = MIMEMultipart()
            msg["From"] = self.email_config["from"]
            msg["To"] = self.email_config["to"]
            msg["Subject"] = f"Ötüken3D Alert: {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity}
            Message: {alert.message}
            Current Value: {value}
            Threshold: {alert.threshold}
            Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.email_config["smtp_server"]) as server:
                server.starttls()
                server.login(
                    self.email_config["username"],
                    self.email_config["password"]
                )
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    def process_metrics(
        self,
        metrics: Dict[str, float]
    ):
        """Process metrics and trigger alerts if needed."""
        current_time = time.time()
        
        for alert in self.alerts.values():
            if alert.name in metrics:
                value = metrics[alert.name]
                
                if self.check_condition(alert, value):
                    if self.should_trigger(alert, current_time):
                        logger.warning(
                            f"Alert triggered: {alert.name}, "
                            f"value: {value}, threshold: {alert.threshold}"
                        )
                        self.send_email_alert(alert, value)
                        self.last_triggered[alert.name] = current_time

class MonitoringService:
    """Complete monitoring service."""
    
    def __init__(
        self,
        prometheus_port: int = 9090,
        collection_interval: int = 60,
        config_path: Optional[Path] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager(config_path, email_config)
        self.collection_interval = collection_interval
        self.running = False
        self.metrics_queue = queue.Queue()
        
        # Start Prometheus server
        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")
    
    def update_prometheus_metrics(
        self,
        metrics: Dict[str, float]
    ):
        """Update Prometheus metrics."""
        try:
            for name, value in metrics.items():
                if name in METRICS:
                    METRICS[name].set(value)
                    
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {str(e)}")
    
    def collect_metrics_loop(self):
        """Continuous metrics collection loop."""
        while self.running:
            try:
                metrics = self.collector.collect_system_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {str(e)}")
    
    def process_metrics_loop(self):
        """Continuous metrics processing loop."""
        while self.running:
            try:
                metrics = self.metrics_queue.get(timeout=1)
                self.update_prometheus_metrics(metrics)
                self.alert_manager.process_metrics(metrics)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Metrics processing failed: {str(e)}")
    
    def start(self):
        """Start monitoring service."""
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self.collect_metrics_loop
        )
        self.collection_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_metrics_loop
        )
        self.processing_thread.start()
        
        logger.info("Monitoring service started")
    
    def stop(self):
        """Stop monitoring service."""
        self.running = False
        self.collection_thread.join()
        self.processing_thread.join()
        logger.info("Monitoring service stopped")

def create_monitoring_service(
    config_path: Optional[Path] = None,
    email_config: Optional[Dict[str, str]] = None
) -> MonitoringService:
    """Create and configure monitoring service."""
    try:
        service = MonitoringService(
            config_path=config_path,
            email_config=email_config
        )
        service.start()
        return service
        
    except Exception as e:
        raise MonitoringError(f"Failed to create monitoring service: {str(e)}") 
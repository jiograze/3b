"""Data management and versioning system for Ötüken3D."""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import hashlib
import time
from datetime import datetime
import sqlite3
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import threading
import schedule

from ..core.logger import setup_logger
from ..core.exceptions import DataError

logger = setup_logger(__name__)

@dataclass
class BackupConfig:
    """Backup configuration."""
    s3_bucket: str
    aws_access_key: str
    aws_secret_key: str
    backup_interval: str = "1d"  # daily
    retention_days: int = 30
    compression: bool = True

@dataclass
class ValidationRule:
    """Data validation rule."""
    field: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: str = "error"  # error/warning

class DataVersion:
    """Data version management."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.versions_path / "metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load version metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"versions": {}}
    
    def save_metadata(self):
        """Save version metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_version(
        self,
        data_path: Path,
        version_name: str,
        description: str = ""
    ) -> str:
        """Create new data version."""
        try:
            # Generate version hash
            timestamp = datetime.now().isoformat()
            content_hash = self._hash_directory(data_path)
            version_id = f"{version_name}_{content_hash[:8]}"
            
            # Create version directory
            version_path = self.versions_path / version_id
            version_path.mkdir(exist_ok=True)
            
            # Copy data
            shutil.copytree(data_path, version_path / "data", dirs_exist_ok=True)
            
            # Update metadata
            self.metadata["versions"][version_id] = {
                "name": version_name,
                "description": description,
                "timestamp": timestamp,
                "hash": content_hash
            }
            self.save_metadata()
            
            return version_id
            
        except Exception as e:
            raise DataError(f"Failed to create version: {str(e)}")
    
    def _hash_directory(self, path: Path) -> str:
        """Calculate directory content hash."""
        sha256_hash = hashlib.sha256()
        
        for filepath in sorted(path.rglob("*")):
            if filepath.is_file():
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version information."""
        return self.metadata["versions"].get(version_id)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [
            {"id": k, **v}
            for k, v in self.metadata["versions"].items()
        ]
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """Compare two versions."""
        try:
            v1_path = self.versions_path / version_id1 / "data"
            v2_path = self.versions_path / version_id2 / "data"
            
            differences = {
                "added": [],
                "removed": [],
                "modified": []
            }
            
            # Compare files
            v1_files = set(f.relative_to(v1_path) for f in v1_path.rglob("*"))
            v2_files = set(f.relative_to(v2_path) for f in v2_path.rglob("*"))
            
            differences["added"] = list(v2_files - v1_files)
            differences["removed"] = list(v1_files - v2_files)
            
            # Check modified files
            common_files = v1_files & v2_files
            for file in common_files:
                if self._hash_file(v1_path / file) != self._hash_file(v2_path / file):
                    differences["modified"].append(file)
            
            return differences
            
        except Exception as e:
            raise DataError(f"Version comparison failed: {str(e)}")
    
    def _hash_file(self, path: Path) -> str:
        """Calculate file hash."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class DataCleaner:
    """Data cleaning utility."""
    
    def __init__(self, rules_path: Optional[Path] = None):
        self.rules = []
        if rules_path:
            self.load_rules(rules_path)
    
    def load_rules(self, rules_path: Path):
        """Load cleaning rules from YAML."""
        with open(rules_path) as f:
            rules_config = yaml.safe_load(f)
            self.rules = [
                ValidationRule(**rule)
                for rule in rules_config["rules"]
            ]
    
    def clean_data(
        self,
        data: pd.DataFrame,
        rules: Optional[List[ValidationRule]] = None
    ) -> pd.DataFrame:
        """Clean data based on rules."""
        try:
            rules = rules or self.rules
            cleaned_data = data.copy()
            
            for rule in rules:
                if rule.rule_type == "remove_duplicates":
                    cleaned_data = cleaned_data.drop_duplicates(
                        subset=rule.parameters.get("columns")
                    )
                
                elif rule.rule_type == "fill_missing":
                    method = rule.parameters.get("method", "mean")
                    if method == "mean":
                        cleaned_data[rule.field].fillna(
                            cleaned_data[rule.field].mean(),
                            inplace=True
                        )
                    elif method == "median":
                        cleaned_data[rule.field].fillna(
                            cleaned_data[rule.field].median(),
                            inplace=True
                        )
                    elif method == "mode":
                        cleaned_data[rule.field].fillna(
                            cleaned_data[rule.field].mode()[0],
                            inplace=True
                        )
                    elif method == "constant":
                        cleaned_data[rule.field].fillna(
                            rule.parameters.get("value"),
                            inplace=True
                        )
                
                elif rule.rule_type == "remove_outliers":
                    std_dev = rule.parameters.get("std_dev", 3)
                    mean = cleaned_data[rule.field].mean()
                    std = cleaned_data[rule.field].std()
                    cleaned_data = cleaned_data[
                        (cleaned_data[rule.field] - mean).abs() <= std_dev * std
                    ]
            
            return cleaned_data
            
        except Exception as e:
            raise DataError(f"Data cleaning failed: {str(e)}")

class BackupManager:
    """Backup management system."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_secret_key
        )
        
        # Start backup scheduler
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self.scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run backup scheduler."""
        if self.config.backup_interval == "1d":
            schedule.every().day.at("00:00").do(self.create_backup)
        elif self.config.backup_interval == "1w":
            schedule.every().week.do(self.create_backup)
        elif self.config.backup_interval == "1h":
            schedule.every().hour.do(self.create_backup)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def create_backup(
        self,
        data_path: Path,
        backup_name: Optional[str] = None
    ) -> str:
        """Create new backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = backup_name or f"backup_{timestamp}"
            
            if self.config.compression:
                backup_file = shutil.make_archive(
                    backup_name,
                    "gzip",
                    data_path
                )
            else:
                backup_file = shutil.make_archive(
                    backup_name,
                    "zip",
                    data_path
                )
            
            # Upload to S3
            s3_key = f"backups/{backup_name}"
            self.s3_client.upload_file(
                backup_file,
                self.config.s3_bucket,
                s3_key
            )
            
            # Cleanup local file
            Path(backup_file).unlink()
            
            return s3_key
            
        except Exception as e:
            raise DataError(f"Backup creation failed: {str(e)}")
    
    def restore_backup(
        self,
        backup_key: str,
        restore_path: Path
    ):
        """Restore from backup."""
        try:
            # Download from S3
            local_file = Path(f"temp_{int(time.time())}")
            self.s3_client.download_file(
                self.config.s3_bucket,
                backup_key,
                str(local_file)
            )
            
            # Extract backup
            shutil.unpack_archive(local_file, restore_path)
            
            # Cleanup
            local_file.unlink()
            
        except Exception as e:
            raise DataError(f"Backup restoration failed: {str(e)}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix="backups/"
            )
            
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"]
                }
                for obj in response.get("Contents", [])
            ]
            
        except Exception as e:
            raise DataError(f"Failed to list backups: {str(e)}")
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        try:
            backups = self.list_backups()
            retention_date = datetime.now() - pd.Timedelta(
                days=self.config.retention_days
            )
            
            for backup in backups:
                if backup["last_modified"] < retention_date:
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=backup["key"]
                    )
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")

class DataValidator:
    """Data validation system."""
    
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules
    
    def validate(
        self,
        data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Validate data against rules."""
        try:
            results = {
                "errors": [],
                "warnings": []
            }
            
            for rule in self.rules:
                if rule.rule_type == "not_null":
                    mask = data[rule.field].isnull()
                    if mask.any():
                        message = f"Null values found in {rule.field}"
                        results[rule.severity].append({
                            "field": rule.field,
                            "rule": rule.rule_type,
                            "message": message,
                            "rows": mask[mask].index.tolist()
                        })
                
                elif rule.rule_type == "unique":
                    duplicates = data[rule.field].duplicated()
                    if duplicates.any():
                        message = f"Duplicate values found in {rule.field}"
                        results[rule.severity].append({
                            "field": rule.field,
                            "rule": rule.rule_type,
                            "message": message,
                            "rows": duplicates[duplicates].index.tolist()
                        })
                
                elif rule.rule_type == "range":
                    min_val = rule.parameters.get("min")
                    max_val = rule.parameters.get("max")
                    if min_val is not None:
                        mask = data[rule.field] < min_val
                        if mask.any():
                            message = f"Values below {min_val} found in {rule.field}"
                            results[rule.severity].append({
                                "field": rule.field,
                                "rule": rule.rule_type,
                                "message": message,
                                "rows": mask[mask].index.tolist()
                            })
                    if max_val is not None:
                        mask = data[rule.field] > max_val
                        if mask.any():
                            message = f"Values above {max_val} found in {rule.field}"
                            results[rule.severity].append({
                                "field": rule.field,
                                "rule": rule.rule_type,
                                "message": message,
                                "rows": mask[mask].index.tolist()
                            })
                
                elif rule.rule_type == "regex":
                    pattern = rule.parameters.get("pattern")
                    mask = ~data[rule.field].str.match(pattern)
                    if mask.any():
                        message = f"Values not matching pattern in {rule.field}"
                        results[rule.severity].append({
                            "field": rule.field,
                            "rule": rule.rule_type,
                            "message": message,
                            "rows": mask[mask].index.tolist()
                        })
            
            return results
            
        except Exception as e:
            raise DataError(f"Data validation failed: {str(e)}")

def create_data_manager(
    base_path: Path,
    backup_config: Optional[BackupConfig] = None,
    rules_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Create complete data management system."""
    try:
        managers = {
            "version": DataVersion(base_path),
            "cleaner": DataCleaner(rules_path),
            "validator": DataValidator([])  # Rules should be loaded from config
        }
        
        if backup_config:
            managers["backup"] = BackupManager(backup_config)
        
        return managers
        
    except Exception as e:
        raise DataError(f"Failed to create data manager: {str(e)}") 
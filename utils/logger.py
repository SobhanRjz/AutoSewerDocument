import json
import time
import os
from typing import Dict, Any
from datetime import datetime
from threading import Lock

class ProgressLogger:
    def __init__(self, log_file: str = "progress_log.json"):
        self.log_file = log_file
        self.lock = Lock()
        self.current_state = {
            "start_time": None,
            "last_update": None,
            "end_time": None,
            "status": "not_started",  # not_started, running, completed, error
            "progress": 0,
            "current_stage": "",
            "stages": {},
            "error": None,
            "details": {}
        }
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize or load existing log file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.current_state = json.load(f)
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Save current state to JSON file"""
        with self.lock:
            with open(self.log_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)

    def start_process(self, stages: Dict[str, float]) -> None:
        """Start the logging process with defined stages and their weights"""
        self.current_state.update({
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "running",
            "progress": 0,
            "stages": {stage: {"weight": weight, "progress": 0, "status": "pending"}
                      for stage, weight in stages.items()},
            "current_stage": "",
            "error": None
        })
        self._save_state()

    def update_stage_progress(self, stage: str, progress: float, details: Dict[str, Any] = None) -> None:
        """Update progress for a specific stage"""
        if stage not in self.current_state["stages"]:
            raise ValueError(f"Stage {stage} not found in registered stages")

        self.current_state["current_stage"] = stage
        self.current_state["stages"][stage]["progress"] = min(progress, 100)
        self.current_state["stages"][stage]["status"] = "running"
        
        if details:
            if "details" not in self.current_state["stages"][stage]:
                self.current_state["stages"][stage]["details"] = {}
            self.current_state["stages"][stage]["details"].update(details)

        # Calculate overall progress
        total_progress = 0
        total_weight = sum(stage_info["weight"] for stage_info in self.current_state["stages"].values())
        
        for stage_name, stage_info in self.current_state["stages"].items():
            stage_contribution = (stage_info["progress"] * stage_info["weight"]) / total_weight
            total_progress += stage_contribution

        self.current_state["progress"] = round(total_progress, 2)
        self.current_state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def complete_stage(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Mark a stage as completed"""
        if stage not in self.current_state["stages"]:
            raise ValueError(f"Stage {stage} not found in registered stages")

        self.current_state["stages"][stage]["progress"] = 100
        self.current_state["stages"][stage]["status"] = "completed"
        
        if details:
            if "details" not in self.current_state["stages"][stage]:
                self.current_state["stages"][stage]["details"] = {}
            self.current_state["stages"][stage]["details"].update(details)

        self._save_state()

    def complete_process(self, details: Dict[str, Any] = None) -> None:
        """Mark the entire process as completed"""
        self.current_state["status"] = "completed"
        self.current_state["progress"] = 100
        self.current_state["end_time"] = datetime.now().isoformat()
        if details:
            self.current_state["details"].update(details)
        self._save_state()

    def log_error(self, error_message: str, details: Dict[str, Any] = None) -> None:
        """Log an error"""
        self.current_state["status"] = "error"
        self.current_state["error"] = error_message
        if details:
            self.current_state["details"].update(details)
        self._save_state()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.current_state 
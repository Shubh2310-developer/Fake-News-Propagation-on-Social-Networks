# backend/tests/services/test_simulation_service.py

import pytest
import asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import json

from app.services.simulation_service import SimulationService
from game_theory.simulation import RepeatedGameSimulation, SimulationConfig


@pytest.mark.unit
class TestSimulationService:
    """Test suite for SimulationService business logic."""

    @pytest.fixture
    def simulation_service(self):
        """Create a SimulationService instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SimulationService()
            service.simulation_storage_path = Path(temp_dir)
            yield service

    @pytest.fixture
    def sample_simulation_config(self):
        """Sample simulation configuration for testing."""
        return {
            "network_config": {
                "num_nodes": 50,
                "network_type": "barabasi_albert",
                "attachment_preference": 3,
                "random_seed": 42
            },
            "game_config": {
                "num_rounds": 5,
                "num_spreaders": 3,
                "num_fact_checkers": 2,
                "num_platforms": 1,
                "learning_rate": 0.1,
                "exploration_rate": 0.1,
                "random_seed": 42
            },
            "description": "Test simulation",
            "save_detailed_history": True
        }

    def test_start_simulation_success(self, simulation_service, sample_simulation_config):
        """Test successful simulation start."""
        with patch.object(simulation_service, '_run_simulation_async') as mock_run:
            mock_run.return_value = None  # Async task started

            simulation_id = simulation_service.start_simulation(sample_simulation_config)

            assert isinstance(simulation_id, str)
            assert len(simulation_id) > 0
            mock_run.assert_called_once()

    def test_get_simulation_status_running(self, simulation_service, sample_simulation_config):
        """Test getting status of a running simulation."""
        # Start a simulation first
        with patch.object(simulation_service, '_run_simulation_async'):
            simulation_id = simulation_service.start_simulation(sample_simulation_config)

        # Mock the simulation as running
        mock_status = {
            "simulation_id": simulation_id,
            "status": "running",
            "progress": 0.6,
            "created_at": "2023-01-01T00:00:00",
            "started_at": "2023-01-01T00:01:00",
            "completed_at": None,
            "error": None,
            "summary": {}
        }

        with patch.object(simulation_service, '_get_status_from_storage') as mock_get:
            mock_get.return_value = mock_status

            status = simulation_service.get_simulation_status(simulation_id)

            assert status["simulation_id"] == simulation_id
            assert status["status"] == "running"
            assert status["progress"] == 0.6

    def test_cancel_simulation_success(self, simulation_service, sample_simulation_config):
        """Test successful simulation cancellation."""
        # Start a simulation
        with patch.object(simulation_service, '_run_simulation_async'):
            simulation_id = simulation_service.start_simulation(sample_simulation_config)

        # Mock the cancellation
        with patch.object(simulation_service, '_cancel_running_task') as mock_cancel:
            mock_cancel.return_value = True

            result = simulation_service.cancel_simulation(simulation_id)

            assert result is True
            mock_cancel.assert_called_once_with(simulation_id)
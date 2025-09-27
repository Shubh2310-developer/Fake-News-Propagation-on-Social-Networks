# backend/tests/api/test_simulation.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import uuid


@pytest.mark.api
class TestSimulationAPI:
    """Test suite for simulation API endpoints."""

    def test_run_simulation_success(self, client: TestClient, sample_game_parameters):
        """
        Tests starting a simulation with valid parameters.
        Verifies that a simulation job is successfully created and returns proper ID.
        """
        with patch('app.services.simulation_service.SimulationService.start_simulation') as mock_start:
            mock_simulation_id = str(uuid.uuid4())
            mock_start.return_value = mock_simulation_id

            response = client.post("/api/v1/simulation/run", json=sample_game_parameters)

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "simulation_id" in data
            assert "status" in data
            assert "message" in data

            # Verify response values
            assert data["status"] == "started"
            assert data["simulation_id"] == mock_simulation_id

    def test_run_simulation_invalid_network_size(self, client: TestClient):
        """
        Tests starting a simulation with invalid network size.
        Verifies that proper validation errors are returned.
        """
        invalid_params = {
            "network_config": {
                "num_nodes": -50,  # Invalid negative value
                "network_type": "barabasi_albert"
            },
            "game_config": {
                "num_rounds": 10,
                "num_spreaders": 5,
                "num_fact_checkers": 3,
                "num_platforms": 1
            }
        }

        response = client.post("/api/v1/simulation/run", json=invalid_params)
        assert response.status_code == 422  # Unprocessable Entity

    def test_get_simulation_status_success(self, client: TestClient):
        """
        Tests getting the status of an existing simulation.
        """
        simulation_id = str(uuid.uuid4())

        with patch('app.services.simulation_service.SimulationService.get_simulation_status') as mock_status:
            mock_status.return_value = {
                "simulation_id": simulation_id,
                "status": "running",
                "progress": 0.5,
                "created_at": "2023-01-01T00:00:00",
                "started_at": "2023-01-01T00:01:00",
                "completed_at": None,
                "error": None,
                "summary": {}
            }

            response = client.get(f"/api/v1/simulation/status/{simulation_id}")

            assert response.status_code == 200
            data = response.json()

            assert "simulation_id" in data
            assert "status" in data
            assert "progress" in data
            assert data["simulation_id"] == simulation_id
            assert data["status"] == "running"
            assert data["progress"] == 0.5

    def test_get_simulation_status_not_found(self, client: TestClient):
        """
        Tests getting the status of a non-existent simulation.
        """
        nonexistent_id = str(uuid.uuid4())

        with patch('app.services.simulation_service.SimulationService.get_simulation_status') as mock_status:
            mock_status.side_effect = ValueError(f"Simulation {nonexistent_id} not found")

            response = client.get(f"/api/v1/simulation/status/{nonexistent_id}")
            assert response.status_code == 404

    def test_cancel_simulation_success(self, client: TestClient):
        """
        Tests cancelling a running simulation.
        """
        simulation_id = str(uuid.uuid4())

        with patch('app.services.simulation_service.SimulationService.cancel_simulation') as mock_cancel:
            mock_cancel.return_value = True

            response = client.post(f"/api/v1/simulation/cancel/{simulation_id}")

            assert response.status_code == 200
            data = response.json()

            assert "message" in data
            assert "simulation_id" in data
            assert "status" in data
            assert data["status"] == "cancelled"
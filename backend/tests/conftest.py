# backend/tests/conftest.py

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.services.classifier_service import ClassifierService
from app.services.simulation_service import SimulationService
from app.services.equilibrium_service import EquilibriumService
from app.services.network_service import NetworkService
from app.services.data_service import DataService
from app.core.database import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Provides a TestClient instance for making API requests in tests.
    This fixture creates a test client that can be used to send HTTP requests
    to the FastAPI application without running a live server.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
async def test_db_engine():
    """
    Create a test database engine using SQLite in-memory database.
    This provides fast, isolated database testing.
    """
    # Use SQLite in-memory database for testing
    database_url = "sqlite+aiosqlite:///:memory:"

    engine = create_async_engine(
        database_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provides a database session for tests with automatic rollback.
    Each test gets a fresh session that's rolled back after the test.
    """
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="module")
def mock_classifier_service() -> ClassifierService:
    """
    Provides a mock ClassifierService with lightweight test models.
    This avoids loading heavy ML models during testing.
    """
    service = ClassifierService()

    # Mock the models with simple test implementations
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.7, 0.3]]  # real=0.7, fake=0.3
    mock_model.is_trained = True
    mock_model.get_model_info.return_value = {"type": "mock", "version": "test"}

    service.models = {
        "ensemble": mock_model,
        "bert": mock_model,
        "logistic_regression": mock_model,
        "lstm": mock_model
    }

    return service


@pytest.fixture(scope="module")
def simulation_service() -> SimulationService:
    """
    Provides a SimulationService instance for testing.
    Uses temporary directory for simulation storage.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        service = SimulationService()
        service.simulation_storage_path = Path(temp_dir)
        yield service


@pytest.fixture(scope="module")
def equilibrium_service() -> EquilibriumService:
    """
    Provides an EquilibriumService instance for testing.
    """
    return EquilibriumService()


@pytest.fixture(scope="module")
def network_service() -> NetworkService:
    """
    Provides a NetworkService instance for testing.
    Uses temporary directory for network storage.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        service = NetworkService()
        service.networks_storage_path = Path(temp_dir)
        yield service


@pytest.fixture(scope="module")
def data_service() -> DataService:
    """
    Provides a DataService instance for testing.
    Uses temporary directory for data storage.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        service = DataService()
        service.data_storage_path = Path(temp_dir)
        yield service


@pytest.fixture
def sample_text_data():
    """
    Provides sample text data for classification testing.
    """
    return [
        {
            "text": "Scientists at MIT have developed a new renewable energy technology.",
            "expected_label": "real"
        },
        {
            "text": "BREAKING: Aliens have landed and world leaders are hiding the truth!",
            "expected_label": "fake"
        },
        {
            "text": "New study shows correlation between exercise and mental health.",
            "expected_label": "real"
        },
        {
            "text": "Doctors HATE this one weird trick that cures everything!",
            "expected_label": "fake"
        }
    ]


@pytest.fixture
def sample_game_parameters():
    """
    Provides sample game theory parameters for testing.
    """
    return {
        "network_config": {
            "num_nodes": 100,
            "network_type": "barabasi_albert",
            "attachment_preference": 3,
            "random_seed": 42
        },
        "game_config": {
            "num_rounds": 10,
            "num_spreaders": 5,
            "num_fact_checkers": 3,
            "num_platforms": 1,
            "learning_rate": 0.1,
            "exploration_rate": 0.1,
            "random_seed": 42
        },
        "description": "Test simulation",
        "save_detailed_history": True
    }


@pytest.fixture
def sample_equilibrium_parameters():
    """
    Provides sample parameters for equilibrium calculations.
    """
    return {
        "network_size": 100,
        "detection_capability": 1.0,
        "max_strategies_per_player": 3,
        "payoff_weights": {
            "spreader": {
                "reach": 1.0,
                "detection_penalty": 0.5,
                "reputation_cost": 0.3
            },
            "fact_checker": {
                "accuracy": 1.0,
                "effort_cost": 0.4,
                "reputation_gain": 0.6
            },
            "platform": {
                "user_engagement": 1.0,
                "moderation_cost": 0.3,
                "credibility_score": 0.5,
                "regulatory_risk": 0.2
            }
        }
    }


@pytest.fixture
def sample_network_config():
    """
    Provides sample network configuration for testing.
    """
    return {
        "network_type": "barabasi_albert",
        "num_nodes": 50,
        "attachment_preference": 3,
        "random_seed": 42
    }


@pytest.fixture
def sample_dataset():
    """
    Provides a sample dataset for data service testing.
    """
    import pandas as pd

    data = {
        "text": [
            "Climate change is a serious environmental issue.",
            "Breaking: Secret government conspiracy revealed!",
            "New medical research published in Nature journal.",
            "You won't believe this miracle cure doctors don't want you to know!",
            "Local university announces new computer science program."
        ],
        "label": [0, 1, 0, 1, 0]  # 0=real, 1=fake
    }

    return pd.DataFrame(data)


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services():
    """
    Automatically mock external services to prevent real API calls during testing.
    """
    with patch('app.core.database.init_db'), \
         patch('app.core.database.close_db'), \
         patch('app.core.cache.init_redis'), \
         patch('app.core.cache.close_redis'):
        yield


@pytest.fixture
def mock_model_loading():
    """
    Mock the model loading process to speed up tests.
    """
    with patch.object(ClassifierService, 'load_models') as mock_load:
        mock_load.return_value = {
            "ensemble": True,
            "bert": True,
            "logistic_regression": True,
            "lstm": True
        }
        yield mock_load


# Utility fixtures for common test patterns
@pytest.fixture
def assert_valid_response():
    """
    Helper fixture for asserting API response validity.
    """
    def _assert_valid_response(response, expected_status=200, required_keys=None):
        assert response.status_code == expected_status

        if expected_status == 200 and required_keys:
            data = response.json()
            for key in required_keys:
                assert key in data, f"Required key '{key}' not found in response"

        return response.json() if expected_status == 200 else None

    return _assert_valid_response


@pytest.fixture
def assert_error_response():
    """
    Helper fixture for asserting error responses.
    """
    def _assert_error_response(response, expected_status):
        assert response.status_code == expected_status

        # Check that error response has proper structure
        if expected_status >= 400:
            data = response.json()
            assert "detail" in data, "Error response should contain 'detail' field"

    return _assert_error_response


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """
    Utility fixture for timing test operations.
    """
    import time

    def _timer(threshold_seconds=1.0):
        start_time = time.time()

        def check_performance():
            elapsed = time.time() - start_time
            assert elapsed < threshold_seconds, f"Operation took {elapsed:.2f}s, expected < {threshold_seconds}s"
            return elapsed

        return check_performance

    return _timer


# Configuration for pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    Automatically clean up any test data after each test.
    """
    yield
    # Cleanup logic here if needed
    # For now, using temporary directories handles most cleanup
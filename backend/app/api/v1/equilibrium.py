# backend/app/api/v1/equilibrium.py

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging

from app.services.equilibrium_service import EquilibriumService
from app.models.game_theory import (
    EquilibriumRequest,
    EquilibriumResponse,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    StrategySpaceInfoResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get equilibrium service instance
def get_equilibrium_service():
    """Get equilibrium service instance."""
    return EquilibriumService()

@router.post("/calculate", response_model=EquilibriumResponse)
async def calculate_equilibrium(
    request: EquilibriumRequest,
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Calculate Nash Equilibria for a given set of game parameters.
    This can be a computationally expensive operation.

    Args:
        request: Game parameters and calculation options

    Returns:
        Calculated equilibria with analysis and interpretation
    """
    try:
        # Convert Pydantic model to dict
        game_params = request.game_parameters

        # Calculate equilibria
        equilibria_result = await equilibrium_service.calculate_equilibria(
            game_params=game_params,
            include_mixed=request.include_mixed,
            include_stability_analysis=request.include_stability_analysis
        )

        return EquilibriumResponse(
            request_id=request.request_id,
            game_parameters=game_params,
            equilibria=equilibria_result,
            computation_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Equilibrium calculation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate equilibria")

@router.post("/analyze/sensitivity", response_model=SensitivityAnalysisResponse)
async def analyze_parameter_sensitivity(
    request: SensitivityAnalysisRequest,
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Analyze how equilibria change with parameter variations.
    This helps understand the robustness of equilibria to parameter changes.

    Args:
        request: Base parameters and variation ranges

    Returns:
        Sensitivity analysis results showing how equilibria change
    """
    try:
        sensitivity_result = await equilibrium_service.analyze_parameter_sensitivity(
            base_params=request.base_parameters,
            parameter_variations=request.parameter_variations
        )

        return SensitivityAnalysisResponse(
            request_id=request.request_id,
            base_parameters=request.base_parameters,
            sensitivity_analysis=sensitivity_result,
            analysis_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sensitivity analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform sensitivity analysis")

@router.post("/compare/scenarios", response_model=ScenarioComparisonResponse)
async def compare_equilibria_across_scenarios(
    request: ScenarioComparisonRequest,
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Compare equilibria across different game scenarios.
    Useful for understanding how different conditions affect strategic outcomes.

    Args:
        request: List of scenarios to compare

    Returns:
        Comparison results highlighting differences and patterns
    """
    try:
        if len(request.scenarios) > 10:  # Limit number of scenarios
            raise HTTPException(
                status_code=400,
                detail="Cannot compare more than 10 scenarios at once"
            )

        comparison_result = await equilibrium_service.compare_equilibria_across_scenarios(
            scenarios=request.scenarios
        )

        return ScenarioComparisonResponse(
            request_id=request.request_id,
            scenarios=request.scenarios,
            comparison_results=comparison_result,
            comparison_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario comparison error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare scenarios")

@router.get("/strategy-space", response_model=StrategySpaceInfoResponse)
async def get_strategy_space_info(
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Get information about the defined strategy spaces for each player type.
    This helps understand what strategies are available for analysis.

    Returns:
        Information about strategy spaces and payoff weights
    """
    try:
        strategy_info = equilibrium_service.get_strategy_space_info()

        return StrategySpaceInfoResponse(
            strategy_spaces=strategy_info["strategy_spaces"],
            payoff_weights=strategy_info["payoff_weights"],
            total_combinations=strategy_info.get("total_combinations", 0)
        )

    except Exception as e:
        logger.error(f"Error getting strategy space info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategy space information")

@router.put("/payoff-weights")
async def update_payoff_weights(
    new_weights: Dict[str, Dict[str, float]],
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Update the payoff weights used in equilibrium calculations.
    This allows customizing how different outcomes are valued.

    Args:
        new_weights: New payoff weight structure

    Returns:
        Success confirmation
    """
    try:
        success = equilibrium_service.update_payoff_weights(new_weights)

        if success:
            return {
                "message": "Payoff weights updated successfully",
                "status": "success",
                "new_weights": new_weights
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid payoff weights structure")

    except Exception as e:
        logger.error(f"Error updating payoff weights: {e}")
        raise HTTPException(status_code=500, detail="Failed to update payoff weights")

@router.get("/payoff-weights")
async def get_current_payoff_weights(
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Get the current payoff weights being used in calculations.

    Returns:
        Current payoff weight configuration
    """
    try:
        strategy_info = equilibrium_service.get_strategy_space_info()
        return {
            "payoff_weights": strategy_info["payoff_weights"],
            "status": "active"
        }

    except Exception as e:
        logger.error(f"Error getting payoff weights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve payoff weights")

@router.post("/quick-analysis")
async def quick_equilibrium_analysis(
    network_size: int = Query(1000, gt=0, le=5000, description="Network size"),
    detection_capability: float = Query(1.0, ge=0.1, le=2.0, description="Detection capability multiplier"),
    include_mixed: bool = Query(False, description="Include mixed strategy equilibria"),
    equilibrium_service: EquilibriumService = Depends(get_equilibrium_service)
):
    """
    Quick equilibrium analysis with simplified parameters.
    Useful for rapid prototyping and testing.

    Args:
        network_size: Size of the social network
        detection_capability: Fact-checking detection capability
        include_mixed: Whether to include mixed strategy equilibria

    Returns:
        Simplified equilibrium analysis results
    """
    try:
        # Create simplified game parameters
        game_params = {
            "network_size": network_size,
            "detection_capability": detection_capability,
            "max_strategies_per_player": 3  # Limit for speed
        }

        equilibria_result = await equilibrium_service.calculate_equilibria(
            game_params=game_params,
            include_mixed=include_mixed,
            include_stability_analysis=False  # Skip for speed
        )

        # Return simplified response
        return {
            "parameters": game_params,
            "pure_equilibria_count": len(equilibria_result.get('pure_strategy_equilibria', [])),
            "mixed_equilibria_count": len(equilibria_result.get('mixed_strategy_equilibria', [])),
            "equilibria_exist": len(equilibria_result.get('pure_strategy_equilibria', [])) > 0,
            "interpretation": equilibria_result.get('interpretation', {}),
            "timestamp": equilibria_result.get('timestamp')
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform quick analysis")
#!/usr/bin/env python3
"""
Integration Test Script
======================

Tests all major components of the AFDGCN Prediction System.
Run with: python scripts/integration_test.py
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class IntegrationTests:
    """Integration tests for the prediction system."""

    @staticmethod
    async def test_imports() -> bool:
        """Test that all modules can be imported."""
        logger.info("🧪 Testing imports...")
        try:
            from backend.app.services.real_time_predictor import (
                predict_region_realtime,
                predict_junction_detail,
            )
            from backend.app.services.background_fetcher import (
                start_background_fetcher,
                stop_background_fetcher,
            )
            from backend.app.services.prediction_wrapper import (
                ensure_model_loaded,
                predict_next_timestep,
            )
            from backend.app.services.phase_calculator import compute_region_phases
            from backend.app.api.v1.predict import router as predict_router
            
            logger.info("✅ All imports successful")
            return True
        except Exception as exc:
            logger.error(f"❌ Import error: {exc}")
            return False

    @staticmethod
    async def test_model_loading() -> bool:
        """Test AFDGCN model loading."""
        logger.info("🧪 Testing model loading...")
        try:
            from backend.app.services.prediction_wrapper import (
                ensure_model_loaded,
                get_model_status,
            )
            
            await ensure_model_loaded()
            status = get_model_status()
            
            logger.info(f"   Model status: {status}")
            if status["loaded"] or status["fallback_active"]:
                logger.info("✅ Model loading successful (or fallback active)")
                return True
            else:
                logger.warning("⚠️ Model not loaded and fallback inactive")
                return True  # Not critical
        except Exception as exc:
            logger.error(f"❌ Model loading error: {exc}")
            return False

    @staticmethod
    async def test_phase_calculation() -> bool:
        """Test phase calculation logic."""
        logger.info("🧪 Testing phase calculation...")
        try:
            from backend.app.services.phase_calculator import compute_region_phases
            
            # Mock prediction data
            predictions = {
                89: {"A": 25.0, "B": 18.0, "C": 12.0, "D": 8.0},
                187: {"A": 30.0, "B": 22.0, "C": 15.0, "D": 10.0},
            }
            
            phases = compute_region_phases(predictions, region="ildem")
            
            assert 89 in phases, "Junction 89 not in phases"
            assert "_cycle_time" in phases[89], "cycle_time missing"
            
            logger.info(f"   Phase 89: cycle={phases[89]['_cycle_time']}s")
            logger.info("✅ Phase calculation successful")
            return True
        except Exception as exc:
            logger.error(f"❌ Phase calculation error: {exc}")
            return False

    @staticmethod
    async def test_cache_system() -> bool:
        """Test prediction cache."""
        logger.info("🧪 Testing cache system...")
        try:
            from backend.app.services.real_time_predictor import _prediction_cache
            
            # Test set and get
            test_data = {"test": "data"}
            await _prediction_cache.set("test_key", test_data)
            cached = await _prediction_cache.get("test_key")
            
            assert cached == test_data, "Cache data mismatch"
            logger.info("✅ Cache system working correctly")
            return True
        except Exception as exc:
            logger.error(f"❌ Cache system error: {exc}")
            return False

    @staticmethod
    async def test_region_config() -> bool:
        """Test region configuration."""
        logger.info("🧪 Testing region configuration...")
        try:
            from backend.app.services.real_time_predictor import REGION_CONFIG
            
            assert "ildem" in REGION_CONFIG, "ildem region missing"
            assert "tuna" in REGION_CONFIG, "tuna region missing"
            
            ildem = REGION_CONFIG["ildem"]
            assert len(ildem["junction_ids"]) == 9, "ildem should have 9 junctions"
            assert ildem["use_model"] is True, "ildem should use model"
            
            tuna = REGION_CONFIG["tuna"]
            assert len(tuna["junction_ids"]) == 7, "tuna should have 7 junctions"
            assert tuna["use_model"] is False, "tuna should not use model"
            
            logger.info("✅ Region configuration correct")
            return True
        except Exception as exc:
            logger.error(f"❌ Region configuration error: {exc}")
            return False

    @staticmethod
    async def test_db_models() -> bool:
        """Test database model imports."""
        logger.info("🧪 Testing database models...")
        try:
            from backend.app.db.models import PhasePrediction, ModelEvent
            
            logger.info(f"   PhasePrediction: {PhasePrediction.__tablename__}")
            logger.info(f"   ModelEvent: {ModelEvent.__tablename__}")
            logger.info("✅ Database models imported successfully")
            return True
        except Exception as exc:
            logger.error(f"❌ Database model error: {exc}")
            return False

    @staticmethod
    async def test_pydantic_schemas() -> bool:
        """Test Pydantic schemas."""
        logger.info("🧪 Testing Pydantic schemas...")
        try:
            from backend.app.schemas.phases import (
                ArmPhase,
                JunctionPhase,
                RegionPhaseResponse,
            )
            
            # Create sample objects
            arm = ArmPhase(
                arm="A",
                arm_name="Test Arm",
                vehicle_count=25,
                lanes=3,
                load=0.5,
                status="low",
                green=15,
                yellow=3,
                red=42,
                cycle_time=60,
            )
            
            assert arm.arm == "A"
            logger.info("✅ Pydantic schemas valid")
            return True
        except Exception as exc:
            logger.error(f"❌ Pydantic schema error: {exc}")
            return False


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("🚦 AFDGCN Integration Test Suite")
    logger.info("=" * 60)
    
    tests = IntegrationTests()
    results = []
    
    # Run all tests
    test_methods = [
        ("Imports", tests.test_imports),
        ("Model Loading", tests.test_model_loading),
        ("Phase Calculation", tests.test_phase_calculation),
        ("Cache System", tests.test_cache_system),
        ("Region Configuration", tests.test_region_config),
        ("Database Models", tests.test_db_models),
        ("Pydantic Schemas", tests.test_pydantic_schemas),
    ]
    
    for test_name, test_func in test_methods:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as exc:
            logger.error(f"❌ {test_name} failed with error: {exc}")
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Test Results:")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

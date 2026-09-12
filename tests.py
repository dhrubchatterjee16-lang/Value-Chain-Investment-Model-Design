import unittest
import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from master_foolproof_ai_engine import (
    MasterValueChain,
    RealOptionsValuationEngine,
    CarbonArbitrageSolver,
    DualLoopClosedLoopCoupling,
    KanderTechnologyAdjustedCBAMEngine,
    ShapleyValueCoalitionEngine,
    BilateralClimateContractEngine
)
from leak_detector import is_anomaly

class TestAllRecommendedPapers(unittest.TestCase):

    def test_shared_responsibility_proportional(self):
        vc = MasterValueChain()
        res = vc.compute_shared_responsibility(mode="proportional")
        self.assertEqual(res['P_total_tCO2e'], 23.0)

    def test_kander_tech_adjusted_cbam(self):
        kander = KanderTechnologyAdjustedCBAMEngine()
        res = kander.compute_adjusted_cbam(base_cbam_tariff=45.0, node_trl=9, abatement_pct=0.60)
        self.assertLess(res['adjusted_cbam_tariff_$/t'], 45.0)
        self.assertGreater(res['technology_discount_pct'], 0)

    def test_shapley_value_coalition(self):
        vc = MasterValueChain()
        shapley = ShapleyValueCoalitionEngine()
        res = shapley.compute_shapley_values(vc, total_coalition_surplus=165.0)
        self.assertEqual(len(res['shapley_allocations']), 6)
        total_allocated = sum(a['shapley_allocated_surplus_$k'] for a in res['shapley_allocations'])
        self.assertAlmostEqual(total_allocated, 165.0, delta=0.2)

    def test_bilateral_climate_contract(self):
        contract = BilateralClimateContractEngine()
        res = contract.generate_contract_clauses()
        self.assertIn('contract_id', res)
        self.assertEqual(res['status'], 'EXECUTED & AUDITED ON CBAM CARBON PASSPORT')

    def test_real_options_jump_diffusion(self):
        rov = RealOptionsValuationEngine()
        eval_res = rov.evaluate_investment(capex=73.5, base_annual_cashflow=18.5, jump_intensity=0.2)
        self.assertGreater(eval_res['jump_adjusted_volatility'], 0.30)

    def test_carbon_arbitrage_solver(self):
        vc = MasterValueChain()
        solver = CarbonArbitrageSolver()
        arb = solver.find_carbon_arbitrage(vc)
        self.assertIn('top_carbon_arbitrage_point', arb)

    def test_leak_detector_is_anomaly(self):
        self.assertTrue(is_anomaly(0.50, 0.35))
        self.assertFalse(is_anomaly(0.20, 0.35))

if __name__ == '__main__':
    unittest.main()

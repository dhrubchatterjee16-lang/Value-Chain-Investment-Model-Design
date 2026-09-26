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
from data_ingestion import CommodityMarketIngestion, SCADA_IoT_Streamer
from leak_detector import is_anomaly

class TestCompleteEngineV9(unittest.TestCase):

    def test_live_commodity_market_ingestion(self):
        mkt = CommodityMarketIngestion()
        data = mkt.fetch_live_prices()
        self.assertIn('brent_crude_$/bbl', data)
        self.assertIn('carbon_ets_$/tCO2e', data)
        self.assertGreater(data['brent_crude_$/bbl'], 0)

    def test_scada_iot_streamer(self):
        streamer = SCADA_IoT_Streamer()
        packet = streamer.fetch_scada_packet(simulate_leak=True)
        self.assertIn('OGMP 2.0 Level 5 Stream', packet['protocol'])
        self.assertEqual(packet['pytorch_tensor'].shape, (1, 6))

    def test_shared_responsibility_proportional(self):
        vc = MasterValueChain()
        res = vc.compute_shared_responsibility(mode="proportional")
        self.assertEqual(res['P_total_tCO2e'], 23.0)

    def test_kander_tech_adjusted_cbam(self):
        kander = KanderTechnologyAdjustedCBAMEngine()
        res = kander.compute_adjusted_cbam(base_cbam_tariff=45.0, node_trl=9, abatement_pct=0.60)
        self.assertLess(res['adjusted_cbam_tariff_$/t'], 45.0)

    def test_shapley_value_coalition(self):
        vc = MasterValueChain()
        shapley = ShapleyValueCoalitionEngine()
        res = shapley.compute_shapley_values(vc, total_coalition_surplus=165.0)
        self.assertEqual(len(res['shapley_allocations']), 6)

    def test_real_options_jump_diffusion(self):
        rov = RealOptionsValuationEngine()
        eval_res = rov.evaluate_investment(capex=73.5, base_annual_cashflow=18.5, jump_intensity=0.2)
        self.assertGreater(eval_res['jump_adjusted_volatility'], 0.30)

    def test_leak_detector_is_anomaly(self):
        self.assertTrue(is_anomaly(0.50, 0.35))
        self.assertFalse(is_anomaly(0.20, 0.35))

    def test_solidity_smart_contract_exporter_v11(self):
        from master_foolproof_ai_engine import SoliditySmartContractExporter
        exporter = SoliditySmartContractExporter()
        sol_code = exporter.generate_solidity_code()
        self.assertIn("contract OilGasSAFClimateContract", sol_code)
        self.assertIn("SPDX-License-Identifier: MIT", sol_code)

    def test_neo4j_cypher_script_exporter_v11(self):
        from gvc_graph import GraphValueChainNetwork
        gvc = GraphValueChainNetwork()
        cypher = gvc.export_neo4j_cypher_script()
        self.assertIn("CREATE", cypher)
        self.assertIn(":GVCNode", cypher)
    def test_physics_informed_autoencoder_pinn(self):
        import torch
        from master_foolproof_ai_engine import PhysicsInformedAutoencoder
        pinn = PhysicsInformedAutoencoder()
        pinn.eval()
        x = torch.tensor([[5.0, 1.0, 8.0, 6.0, 2.0, 1.0]])
        recon = pinn(x)
        self.assertEqual(recon.shape, (1, 6))
        loss_dict = pinn.compute_pinn_loss(x, recon)
        self.assertIn('total_pinn_loss', loss_dict)
        self.assertIn('mass_balance_loss', loss_dict)
        self.assertIn('stoich_loss', loss_dict)
        self.assertIn('thermo_enthalpy_loss', loss_dict)

    def test_value_chain_gcn(self):
        import torch
        from master_foolproof_ai_engine import ValueChainGCN
        gcn = ValueChainGCN()
        gcn.eval()
        node_features = torch.randn(10, 4)
        adj = torch.eye(10)
        emb = gcn(node_features, adj)
        self.assertEqual(emb.shape, (1, 10, 8))

    def test_multi_agent_ppo_policy_mappo(self):
        import torch
        from master_foolproof_ai_engine import MultiAgentPPOPolicy
        mappo = MultiAgentPPOPolicy()
        joint_states = torch.randn(2, 6, 4)
        actions, values = mappo(joint_states)
        self.assertEqual(actions.shape, (2, 6, 2))
        self.assertEqual(values.shape, (2, 1))
        
        old_log_probs = torch.zeros(2, 6, 2)
        new_log_probs = torch.log(actions + 1e-6)
        advantages = torch.ones(2, 1)
        next_values = values + 0.1
        loss_dict = mappo.compute_ppo_surrogate_loss(old_log_probs, new_log_probs, advantages, state_values=values, next_state_values=next_values)
        self.assertIn('total_ppo_loss', loss_dict)
        self.assertIn('hjb_residual_loss', loss_dict)

    def test_advanced_math_and_economic_engines(self):
        from master_foolproof_ai_engine import (
            MultiRegionInputOutputEngine,
            ContinuousTimeHJBGameSolver,
            PigouvianGeneralEquilibriumSolver,
            KuhnTuckerCapitalAllocationSolver,
            AtkinsonWelfareEquityEngine
        )
        
        # 1. MRIO Test
        mrio = MultiRegionInputOutputEngine()
        mrio_res = mrio.compute_mrio_multipliers()
        self.assertEqual(len(mrio_res['regions']), 3)
        self.assertEqual(len(mrio_res['miyazawa_income_multipliers']), 3)
        
        # 2. HJB Game Solver Test
        hjb = ContinuousTimeHJBGameSolver()
        hjb_res = hjb.solve_hjb_pde()
        self.assertIn('hjb_upstream_val_$k', hjb_res)
        
        # 3. CGE Pigouvian Tax Test
        cge = PigouvianGeneralEquilibriumSolver()
        cge_res = cge.solve_cge_clearing()
        self.assertGreater(cge_res['optimal_pigouvian_tax_t_star_$/bbl'], 0)
        
        # 4. Kuhn-Tucker Solver Test
        kt = KuhnTuckerCapitalAllocationSolver()
        kt_res = kt.solve_kuhn_tucker()
        self.assertIn('kt_optimal_project_indices', kt_res)
        
        # 5. Atkinson Equity Test
        atkinson = AtkinsonWelfareEquityEngine()
        atk_res = atkinson.compute_atkinson_index()
        self.assertGreater(atk_res['distributional_equity_improvement_pct'], 0)

if __name__ == '__main__':
    unittest.main()



from pymgrid.Environments.ScenarioEnvironment import CSPLAScenarioEnvironment
from pymgrid.Microgrid import Microgrid


class RuleBaseControl(CSPLAScenarioEnvironment):
    def __init__(
        self,
        tsStartIndex,
        tsLength,
        env_config,
        customPVTs=None,
        customLoadTs=None,
        pv_factor=1,
        action_design="rule-based",
        Na=6,
        mode="naive",
        seed=42,
    ):
        super().__init__(
            tsStartIndex,
            tsLength,
            env_config,
            customPVTs,
            customLoadTs,
            pv_factor,
            action_design,
            Na,
            mode,
            seed,
        )

    @staticmethod
    def rule_based_policy(self, policy: str = "just_buy", mg: Microgrid = None) -> dict:
        if mg is not None:
            self.mg = mg
        if policy == "just_buy":
            return self.just_buy(self.mg)
        else:
            raise ValueError(f"Unknown policy {policy}")

    @staticmethod
    def just_buy(self, mg) -> dict:
        balance = mg.load - mg.pv
        control_dict = {
            "battery_charge": 0,
            "battery_discharge": 0,
            "grid_import": max(0, balance),
            "grid_export": 0,
            "pv": mg.pv,
            "pv_consummed": mg.pv,
            "genset": 0,
        }
        return control_dict

    @staticmethod
    def buy_sell(self, mg) -> dict:
        balance = mg.load - mg.pv
        control_dict = {
            "battery_charge": 0,
            "battery_discharge": 0,
            "grid_import": max(0, balance),
            "grid_export": -min(0, balance),
            "pv": mg.pv,
            "pv_consummed": mg.pv,
            "genset": 0,
        }
        return control_dict

    @staticmethod
    def buy_store(self, mg) -> dict:
        balance = mg.load - mg.pv
        control_dict = {
            "battery_charge": -min(0, balance),
            "battery_discharge": 0,
            "grid_import": max(0, balance),
            "grid_export": 0,
            "pv": mg.pv,
            "pv_consummed": mg.pv,
            "genset": 0,
        }
        return control_dict

    @staticmethod
    def store(self, mg) -> dict:
        capa_to_charge = mg.battery.capa_to_charge
        balance = mg.load - mg.pv
        control_dict = {
            "battery_charge": capa_to_charge,
            "battery_discharge": 0,
            "grid_import": max(0, balance + capa_to_charge),
            "grid_export": 0,
            "pv": mg.pv,
            "pv_consummed": mg.pv,
            "genset": 0,
        }
        return control_dict

    @staticmethod
    def discharge(self, mg) -> dict:
        capa_to_discharge = mg.battery.capa_to_discharge
        balance = mg.load - mg.pv
        control_dict = {
            "battery_charge": 0,
            "battery_discharge": min(capa_to_discharge, balance),
            "grid_import": max(0, balance - capa_to_discharge),
            "grid_export": 0,
            "pv": mg.pv,
            "pv_consummed": mg.pv,
            "genset": 0,
        }
        return control_dict

    @staticmethod
    def buy_low_discharge_high(self, mg, price_mode="mean") -> dict:
        if price_mode == "mean":
            price_ref = mg._grid_price_import.mean().item()
        elif price_mode == "min":
            price_ref = mg._grid_price_import.min().item()
        else:
            raise ValueError(f"Unknown price mode : {price_mode}")
        if mg._next_grid_price_import < price_ref:
            return self.store(mg)
        else:
            return self.discharge(mg)

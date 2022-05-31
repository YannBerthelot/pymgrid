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
    def just_buy(mg) -> dict:
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
    def buy_sell(mg) -> dict:
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
    def buy_store(mg) -> dict:
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
    def store(mg) -> dict:
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
    def discharge(mg) -> dict:
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
    def buy_low_discharge_high(mg, price_mode="mean") -> dict:
        if price_mode == "mean":
            price_ref = mg._grid_price_import.mean().item()
        elif price_mode == "min":
            price_ref = mg._grid_price_import.min().item()
        else:
            raise ValueError(f"Unknown price mode : {price_mode}")
        if mg._next_grid_price_import < price_ref:
            return RuleBaseControl.store(mg)
        else:
            return RuleBaseControl.discharge(mg)

    @staticmethod
    def rule_based_policy(policy: str = "just_buy", mg: Microgrid = None) -> dict:
        if policy == "just_buy":
            return RuleBaseControl.just_buy(mg)
        else:
            raise ValueError(f"Unknown policy {policy}")

    def step(self, control_dict):
        self.mg.run(control_dict)

        # COMPUTE NEW STATE AND REWARD
        self.state = self.transition()
        self.reward = self.get_reward()
        self.done = self.mg.done
        self.info = {}
        self.round += 1
        # print("reward", self.reward)

        return self.state, self.reward, self.done, self.info

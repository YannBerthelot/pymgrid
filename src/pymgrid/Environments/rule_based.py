import gym
import numpy as np
from pymgrid.Environments.ScenarioEnvironment import CSPLAScenarioEnvironment
import pymgrid.Environments.Environment as pymgridEnvs
from pymgrid.Microgrid import Microgrid


class Policies:
    def __init__(self) -> None:
        self.policies = {
            0: RuleBaseControl.just_buy,
            1: RuleBaseControl.buy_sell,
            2: RuleBaseControl.buy_low_discharge_high,
            3: RuleBaseControl.buy_min_discharge_high,
        }

    @property
    def _Na(self):
        return len(self.policies)

    def get_action(self, action: int, mg):
        """
        Policy orchestration
        """

        return self.policies[int(action)](mg)


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
        self.policies = Policies().policies

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
    def buy_mean_discharge_high(mg) -> dict:
        return RuleBaseControl.buy_low_discharge_high(mg)

    @staticmethod
    def buy_min_discharge_high(mg) -> dict:
        return RuleBaseControl.buy_low_discharge_high(mg, price_mode="min")

    @staticmethod
    def rule_based_policy(policy: str = "just_buy", mg: Microgrid = None) -> dict:
        if policy == "just_buy":
            return RuleBaseControl.just_buy(mg)
        else:
            raise ValueError(f"Unknown policy {policy}")

    def step_RBC(self, control_dict):
        self.mg.run(control_dict)

        # COMPUTE NEW STATE AND REWARD
        self.state = self.transition()
        self.reward = self.get_reward()
        self.done = self.mg.done
        self.info = {}
        self.round += 1
        # print("reward", self.reward)

        return self.state, self.reward, self.done, self.info


class MacroEnvironment(pymgridEnvs.Environment):
    def __init__(self, env_config, microPolicies, switchingFrequency=1, seed=42):
        """
        Input
        list microPolicies -- list of Policy objects implementing method getAction: mg |--> action
        int switchingFrequency -- the frequency of switching micro policies
        """
        print("\nENV", env_config)
        print("\nSEED", seed)
        print("\n")
        super(MacroEnvironment, self).__init__(env_config=env_config, seed=seed)
        self.env_config = env_config
        self.switchingFrequency = switchingFrequency
        self.policies = Policies()
        self.microPolicies = self.policies.policies

        # microPolicy action design
        self.Na = self.policies._Na
        self.TRAIN = True
        self.action_space = gym.spaces.Discrete(self.Na)
        self.reset()

    def reset(self, testing=False):
        if "testing" in self.env_config:
            testing = self.env_config["testing"]
        self.round = 1
        # Reseting microgrid
        self.mg.reset(testing=testing)
        if testing == True:
            self.TRAIN = False
        elif self.resampling_on_reset == True:
            Preprocessing.sample_reset(
                self.mg.architecture["grid"] == 1,
                self.saa,
                self.mg,
                sampling_args=sampling_args,
            )

        self.state, self.reward, self.done, self.info = (
            self.transition(),
            0,
            False,
            {},
        )

        return self.state

    def step(self, action):

        # CONTROL (pymgrid's Native)
        if self.done:
            print("WARNING : EPISODE DONE")  # should never reach this point
            return self.state, self.reward, self.done, self.info
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("ERROR : INVALID STATE", self.state)

        try:
            assert self.action_space.contains(action)
        except AssertionError:
            print("ERROR : INVALD ACTION", action)

        # UPDATE THE MICROGRID (self.switchingFrequency times)
        self.reward = 0

        for i in np.arange(self.switchingFrequency):
            # control_dict = self.micro_policy(self.microPolicies[action].getAction(self))
            control_dict = self.policies.get_action(action, self.mg)
            # print("CD:", control_dict)

            self.mg.run(control_dict)

            # COMPUTE NEW STATE AND REWARD
            self.state = self.transition()
            self.reward += self.get_reward()
            self.done = self.mg.done
            self.info = {}
            self.round += 1

        return self.state, self.reward, self.done, self.info

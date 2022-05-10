import pymgrid.Environments.Environment as pymgridEnvs
import numpy as np
import pandas as pd
import pymgrid.Environments.Preprocessing as Preprocessing
from pymgrid.Environments.MacroEnvironment import RBCPolicy
import gym
from copy import copy


class ScenarioEnvironment(pymgridEnvs.Environment):
    def __init__(
        self,
        tsStarts,
        tsLength,
        env_config,
        customPVTs=None,
        customLoadTs=None,
        pv_factor=1.0,
        seed=42,
    ):
        """
        Input
        int tsStartIndex -- start of the piece of time series
        int tsLength -- length of the piece of the time series to extract starting from tsStartIndex
        dict envConfig -- pymgridEnvs.Environment native dictionary for config
        """
        # Set seed
        np.random.seed(seed)

        self.TRAIN = True  # no reward smoothing if set to false
        # Microgrid
        self.env_config = env_config
        self.mg = copy(env_config["microgrid"])
        self.RBC = RBCPolicy(self.mg)
        self.tsStarts = tsStarts
        self.tsLength = tsLength
        if not (customPVTs is None or customLoadTs is None):
            self.mg._load_ts = pd.DataFrame(
                customLoadTs, columns=["Electricity:Facility [kW](Hourly)"]
            )
            self.mg._pv_ts = pd.DataFrame(customPVTs, columns=["GH illum (lx)"])
        self._pv_ts_initial = pv_factor * self.mg._pv_ts
        self._load_ts_initial = self.mg._load_ts
        self._grid_price_import_initial = self.mg._grid_price_import
        self._grid_price_export_initial = self.mg._grid_price_export
        self._grid_status_ts_initial = self.mg._grid_status_ts
        self._grid_co2_initial = self.mg._grid_co2

        self.set_timeseries(tsStarts[0], tsLength)
        # setting the piece to be the main time series

        # State space

        # self.mg.train_test_split() # we do not need it, use the whole time series

        # Number of states
        self.Ns = len(self.mg._df_record_state.keys()) + 1

        # array of normalizing constants
        self.states_normalization = Preprocessing.normalize_environment_states(
            env_config["microgrid"]
        )

        # training_reward_smoothing
        try:
            self.training_reward_smoothing = env_config["training_reward_smoothing"]
        except:
            self.training_reward_smoothing = "sqrt"

        self.resampling_on_reset = False  # we do not care, anyway it is some internal-training-crucial stuff we do not use

        # setting observation space
        self.observation_space = gym.spaces.Box(
            low=-1, high=np.float("inf"), shape=(self.Ns,), dtype=np.float
        )

        # Action space
        self.metadata = {"render.modes": ["human"]}

        self.state, self.reward, self.done, self.info, self.round = (
            None,
            None,
            None,
            None,
            None,
        )
        self.round = None

        # Start the first round
        self.seed()
        self.reset()

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("ERROR : INVALID STATE", self.state)

    def set_timeseries(self, tsStartIndex, tsLength):
        self.mg._pv_ts = self._pv_ts_initial[tsStartIndex : (tsStartIndex + tsLength)]
        self.mg._load_ts = self._load_ts_initial[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_price_import = self._grid_price_import_initial[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_price_export = self._grid_price_export_initial[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_status_ts = self._grid_status_ts_initial[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_co2 = self._grid_co2_initial[
            tsStartIndex : (tsStartIndex + tsLength)
        ]

    def reset(self, testing=False):
        if "testing" in self.env_config:
            testing = self.env_config["testing"]
        self.round = 1
        start = np.random.choice(self.tsStarts)
        self.set_timeseries(start, self.tsLength)
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


class CSPLAScenarioEnvironment(ScenarioEnvironment):
    def __init__(
        self,
        tsStartIndex,
        tsLength,
        env_config,
        customPVTs=None,
        customLoadTs=None,
        pv_factor=1.0,
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
            seed,
        )

        # cspla action design
        print(f"Action design : {action_design}")
        self.action_design = action_design
        self.mode = mode
        self.Na = (
            2 + self.mg.architecture["grid"] * 3 + self.mg.architecture["genset"] * 1
        )
        if self.mg.architecture["grid"] == 1 and self.mg.architecture["genset"] == 1:
            self.Na += 1
        self.Na = Na
        self.action_space = gym.spaces.Discrete(self.Na)

    def larger_micro_policy_failsafe(self, action):
        mg = self.mg
        load = mg.load
        pv = mg.pv
        net_load = load - pv
        excess = -net_load
        capa_to_charge = mg.battery.capa_to_charge
        capa_to_discharge = mg.battery.capa_to_discharge
        p_charge_max = mg.battery.p_charge_max
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))
        p_charge = max(0, min(excess, capa_to_charge, p_charge_max))
        p_charge_sup = min(capa_to_charge, p_charge_max)
        p_discharge_sup = min(capa_to_discharge, p_discharge_max)
        policies = {
            0: {  # sell_excess
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, net_load),
                "grid_export": max(0, excess),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            1: {  # store_excess
                "battery_charge": p_charge,
                "battery_discharge": 0,
                "grid_import": max(0, net_load),
                "grid_export": max(0, excess - p_charge),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            2: {  # fill_battery_from_grid
                "battery_charge": p_charge_sup,
                "battery_discharge": 0,
                "grid_import": max(0, p_charge_sup + net_load),
                "grid_export": max(0, excess - p_charge_sup),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            3: {  # fill_battery_from_grid_75
                "battery_charge": p_charge_sup * 0.75,
                "battery_discharge": 0,
                "grid_import": max(0, (p_charge_sup * 0.75) + net_load),
                "grid_export": max(0, excess - (p_charge_sup * 0.75)),
                "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            4: {  # fill_battery_from_grid_50
                "battery_charge": p_charge * 0.5,
                "battery_discharge": 0,
                "grid_import": max(0, (p_charge * 0.5) + net_load),
                "grid_export": max(0, excess - (p_charge * 0.5)),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            5: {  # fill_battery_from_grid_25
                "battery_charge": p_charge_sup * 0.25,
                "battery_discharge": 0,
                "grid_import": max(0, (p_charge_sup * 0.25) + net_load),
                "grid_export": max(0, excess - (p_charge_sup * 0.25)),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            6: {  # discharge_to_sell
                "battery_charge": 0,
                "battery_discharge": p_discharge_sup,
                "grid_import": max(0, net_load - p_discharge_sup),
                "grid_export": max(0, excess + p_discharge_sup),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            7: {  # discharge_to_sell_75
                "battery_charge": 0,
                "battery_discharge": p_discharge_sup * 0.75,
                "grid_import": max(0, net_load - (p_discharge_sup * 0.75)),
                "grid_export": max(0, excess + (p_discharge_sup * 0.75)),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            8: {  # discharge_to_sell_50
                "battery_charge": 0,
                "battery_discharge": p_discharge_sup * 0.5,
                "grid_import": max(0, net_load - (p_discharge_sup * 0.5)),
                "grid_export": max(0, excess + (p_discharge_sup * 0.5)),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            9: {  # discharge_to_sell_25
                "battery_charge": 0,
                "battery_discharge": p_discharge_sup * 0.25,
                "grid_import": max(0, net_load - (p_discharge_sup * 0.25)),
                "grid_export": max(0, excess + (p_discharge_sup * 0.25)),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
            10: {  # discharge_for_load
                "battery_charge": 0,
                "battery_discharge": p_discharge,
                "grid_import": max(0, net_load - p_discharge),
                "grid_export": max(0, excess),
                # "pv": pv,
                "pv_consummed": pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            },
        }
        # print(policies[action])
        return policies[action]

    def micro_policy_failsafe(self, action):
        mg = self.mg
        load = mg.load
        pv = mg.pv
        capa_to_charge = mg.battery.capa_to_charge
        capa_to_discharge = mg.battery.capa_to_discharge
        policies = {
            "sell_excess": {  # 0
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": max(0, pv - load),
                "pv": pv,
                "genset": 0,
            },
            "store_excess": {  # 1
                "battery_charge": min(max(0, pv - load), capa_to_charge),
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": max(0, pv - capa_to_charge - load),
                "pv": pv,
                "genset": 0,
            },
            "fill_battery_from_grid": {  # 2
                "battery_charge": capa_to_charge,
                "battery_discharge": 0,
                "grid_import": max(0, capa_to_charge + load - pv),
                "grid_export": max(0, pv - capa_to_charge - load),
                "pv": pv,
                "genset": 0,
            },
            # "fill_battery_from_genset": {  # outdated
            #     "battery_charge": capa_to_charge,
            #     "battery_discharge": 0,
            #     "grid_import": 0,
            #     "grid_export": 0,
            #     "pv": pv,
            #     "genset": max(0, capa_to_charge + (load - pv)),
            # },
            # "buy_for_load": {  # outdated
            #     "battery_charge": 0,
            #     "battery_discharge": 0,
            #     "grid_import": max(0, load - pv),
            #     "grid_export": 0,
            #     "pv": min(pv, load),
            #     "genset": 0,
            # },
            # "genset_for_load": {  # outdated
            #     "battery_charge": 0,
            #     "battery_discharge": 0,
            #     "grid_import": 0,
            #     "grid_export": 0,
            #     "pv": min(pv, load),
            #     "genset": max(0, load - pv),
            # },
            "discharge_to_sell": {  # 3
                "battery_charge": 0,
                "battery_discharge": capa_to_discharge,
                "grid_import": max(0, load - pv - capa_to_discharge),
                "grid_export": max(0, pv + capa_to_discharge - load),
                "pv": pv,
                "genset": 0,
            },
            "discharge_for_load": {  # 4
                "battery_charge": 0,
                "battery_discharge": min(capa_to_discharge, max(0, load - pv)),
                "grid_import": max(
                    0, load - pv - min(capa_to_discharge, max(0, load - pv))
                ),
                "grid_export": max(
                    0, pv + min(capa_to_discharge, max(0, load - pv)) - load
                ),
                "pv": pv,
                "genset": 0,
            },
        }
        return policies[list(set(policies.keys()))[action]]

    def micro_policy(self, action):
        mg = self.mg
        load = mg.load
        pv = mg.pv
        capa_to_charge = mg.battery.capa_to_charge
        capa_to_discharge = mg.battery.capa_to_discharge
        policies = {
            "sell_excess": {  # 0
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": max(0, pv - load),
                "pv": pv,
                "genset": 0,
            },
            "store_excess": {  # 1
                "battery_charge": min(max(0, pv - load), capa_to_charge),
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": max(0, pv - capa_to_charge - load),
                "pv": pv,
                "genset": 0,
            },
            "fill_battery_from_grid": {  # 2
                "battery_charge": capa_to_charge,
                "battery_discharge": 0,
                "grid_import": max(0, capa_to_charge + load - pv),
                "grid_export": 0,
                "pv": pv,
                "genset": 0,
            },
            "discharge_to_sell": {  # 3
                "battery_charge": 0,
                "battery_discharge": capa_to_discharge,
                "grid_import": 0,
                "grid_export": max(0, pv + capa_to_discharge - load),
                "pv": pv,
                "genset": 0,
            },
            "discharge_for_load": {  # 4
                "battery_charge": 0,
                "battery_discharge": min(capa_to_discharge, max(0, load - pv)),
                "grid_import": max(
                    0, load - pv - min(capa_to_discharge, max(0, load - pv))
                ),
                "grid_export": 0,
                "pv": pv,
                "genset": 0,
            },
            "buy_for_load": {  # 5
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": 0,
                "pv": pv,
                "genset": 0,
            },
        }
        return policies[list(set(policies.keys()))[action]]

    def rule_based(self, mode: str = "naive") -> dict:
        """
        Select next action using rule based method

        Args:
            obs (np.array): Array describing the environment observation. Values are : load, pv, battery_soc, capa_to_charge, capa_to_discharge, grid_status, grid_co2, grid_price_import, grid_price_export, hour_sin, hour_cos

        Returns:
            dict: The action dictionnary to be used for next timestep.
        """

        mg = self.mg
        net_load = mg.load - mg.pv
        if mode == "naive":
            if net_load > 0:
                # Lack of energy
                if mg.battery.capa_to_discharge > 0:
                    action_dict = {  # discharge battery as much as possible and buy the missing energy
                        "battery_charge": 0,
                        "battery_discharge": min(
                            min(mg.battery.capa_to_discharge, net_load),
                            mg.battery.p_discharge_max,
                        ),
                        "grid_import": net_load
                        - min(
                            min(mg.battery.capa_to_discharge, net_load),
                            mg.battery.p_discharge_max,
                        ),
                        "grid_export": 0,
                        # "pv": pv,
                        "pv_consummed": mg.pv,
                        "pv_curtailed": 0.0,
                        "genset": 0,
                    }
                else:
                    action_dict = {  # buy missing energy
                        "battery_charge": 0,
                        "battery_discharge": 0,
                        "grid_import": net_load,
                        "grid_export": 0,
                        # "pv": pv,
                        "pv_consummed": mg.pv,
                        "pv_curtailed": 0.0,
                        "genset": 0,
                    }
            else:
                # Excess of energy
                if mg.battery.capa_to_charge > 0:
                    action_dict = {  # store excess
                        "battery_charge": min(
                            min(mg.battery.capa_to_charge, -net_load),
                            mg.battery.p_charge_max,
                        ),
                        "battery_discharge": 0,
                        "grid_import": 0,
                        "grid_export": max(
                            0,
                            -net_load
                            - min(mg.battery.capa_to_charge, mg.battery.p_charge_max),
                        ),
                        # "pv": pv,
                        "pv_consummed": mg.pv,
                        "pv_curtailed": 0.0,
                        "genset": 0,
                    }
                else:
                    action_dict = {  # sell excess
                        "battery_charge": 0,
                        "battery_discharge": 0,
                        "grid_import": 0,
                        "grid_export": -net_load,
                        # "pv": pv,
                        "pv_consummed": mg.pv,
                        "pv_curtailed": 0.0,
                        "genset": 0,
                    }
        elif mode == "basic":
            action_dict = {  # sell excess and import what's needed
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, net_load),
                "grid_export": max(0, -net_load),
                # "pv": pv,
                "pv_consummed": mg.pv,
                "pv_curtailed": 0.0,
                "genset": 0,
            }
        else:
            raise ValueError(f"mode {mode} is not implemented")
        return action_dict

    def get_action(self, action):
        """
        CSPLA action design
        """
        if self.action_design == "large":
            return self.larger_micro_policy_failsafe(action)
        elif self.action_design == "rule-based":
            return self.RBC.getAction(self)
        else:
            return self.micro_policy(action)

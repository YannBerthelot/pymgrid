import pymgrid.Environments.Environment as pymgridEnvs
import numpy as np
import pandas as pd
import pymgrid.Environments.Preprocessing as Preprocessing
import gym
from copy import copy


class ScenarioEnvironment(pymgridEnvs.Environment):
    def __init__(self, tsStarts, tsLength, env_config, seed=42):
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
        self.tsStarts = tsStarts
        self.tsLength = tsLength
        self._pv_ts_initial = self.mg._pv_ts
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
    def __init__(self, tsStartIndex, tsLength, env_config, seed=42):
        super().__init__(tsStartIndex, tsLength, env_config, seed)

        # cspla action design
        self.Na = (
            2 + self.mg.architecture["grid"] * 3 + self.mg.architecture["genset"] * 1
        )
        if self.mg.architecture["grid"] == 1 and self.mg.architecture["genset"] == 1:
            self.Na += 1
        self.Na = 5
        self.action_space = gym.spaces.Discrete(self.Na)

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
            "discharge_to_sell": {  # 4
                "battery_charge": 0,
                "battery_discharge": capa_to_discharge,
                "grid_import": max(0, load - pv - capa_to_discharge),
                "grid_export": max(0, pv + capa_to_discharge - load),
                "pv": pv,
                "genset": 0,
            },
            "discharge_for_load": {  # 5
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

    def get_action(self, action):
        """
        CSPLA action design
        """
        return self.micro_policy(action)

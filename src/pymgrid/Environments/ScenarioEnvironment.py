import pymgrid.Environments.Environment as pymgridEnvs
import numpy as np
import pandas as pd
import pymgrid.Environments.Preprocessing as Preprocessing
import gym


class ScenarioEnvironment(pymgridEnvs.Environment):
    def __init__(self, tsStartIndex, tsLength, env_config, customPVTs=None, customLoadTs=None, seed=42):
        """
        Input
        int tsStartIndex -- start of the piece of time series
        int tsLength -- length of the piece of the time series to extract starting from tsStartIndex
        dict envConfig -- pymgridEnvs.Environment native dictionary for config
        float[][] customPVTS  --  (T,1)-shaped np.array representing pv time series (if None, the native time series is used)  
        float[][] customLoadTS   --  (T,1)-shaped np.array representing load time series (if None, the native time series is used)
        """
        # Set seed
        np.random.seed(seed)

        self.TRAIN = True  # no reward smoothing if set to false

        # Microgrid
        self.env_config = env_config
        self.mg = env_config["microgrid"]
        # setting the piece to be the main time series

        if not (customPVTs is None or customLoadTs is None):
            self.mg._load_ts  = pd.DataFrame(customLoadTs, columns = ["Electricity:Facility [kW](Hourly)"])
            self.mg._pv_ts = pd.DataFrame(customPVTs, columns = ["GH illum (lx)"])

        self.mg._pv_ts = self.mg._pv_ts[tsStartIndex : (tsStartIndex + tsLength)]
        self.mg._load_ts = self.mg._load_ts[tsStartIndex : (tsStartIndex + tsLength)]
        self.mg._grid_price_import = self.mg._grid_price_import[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_price_export = self.mg._grid_price_export[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_status_ts = self.mg._grid_status_ts[
            tsStartIndex : (tsStartIndex + tsLength)
        ]
        self.mg._grid_co2 = self.mg._grid_co2[tsStartIndex : (tsStartIndex + tsLength)]

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


class CSPLAScenarioEnvironment(ScenarioEnvironment):
    def __init__(self, tsStartIndex, tsLength, env_config, customPVTs=None, customLoadTs=None, seed=42):
        super().__init__(tsStartIndex, tsLength, env_config, customPVTs, customLoadTs, seed)

        # cspla action design
        self.Na = (
            2 + self.mg.architecture["grid"] * 3 + self.mg.architecture["genset"] * 1
        )
        if self.mg.architecture["grid"] == 1 and self.mg.architecture["genset"] == 1:
            self.Na += 1
        self.Na = 8
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
                "pv": min(pv, load),
                "genset": 0,
            },
            "store_excess": {  # 1
                "battery_charge": max(0, pv - load),
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "fill_battery_from_grid": {  # 2
                "battery_charge": capa_to_charge,
                "battery_discharge": 0,
                "grid_import": capa_to_charge + (load - pv),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "fill_battery_from_genset": {  # 3
                "battery_charge": capa_to_charge,
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": max(0, capa_to_charge + (load - pv)),
            },
            "buy_for_load": {  # 4
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "genset_for_load": {  # 5
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": max(0, load - pv),
            },
            "discharge_to_sell": {  # 6
                "battery_charge": 0,
                "battery_discharge": capa_to_discharge,
                "grid_import": max(0, load - pv - capa_to_discharge),
                "grid_export": max(0, capa_to_discharge + pv - load),
                "pv": pv,
                "genset": 0,
            },
            "discharge_for_load": {  # 7
                "battery_charge": 0,
                "battery_discharge": min(capa_to_discharge, load - pv),
                "grid_import": max(0, load - pv - capa_to_discharge),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
        }
        return policies[list(set(policies.keys()))[action]]

    def get_action(self, action):
        """
        CSPLA action design
        """
        return self.micro_policy(action)
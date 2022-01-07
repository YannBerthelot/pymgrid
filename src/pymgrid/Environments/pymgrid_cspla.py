"""
Copyright 2020 Total S.A
Authors:Gonzague Henri <gonzague.henri@total.com>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/10/21 07:43 $
Gonzague Henri
"""

"""
<pymgrid is a Python library to simulate microgrids>
Copyright (C) <2020> <Total S.A.>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
from pymgrid.Environments.Environment import Environment
import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box


class MicroGridEnv(Environment):
    """
    Markov Decision Process associated to the microgrid.

        Parameters
        ----------
            microgrid: microgrid, mandatory
                The controlled microgrid.
            random_seed: int, optional
                Seed to be used to generate the needed random numbers to size microgrids.

    """

    def __init__(self, env_config, seed=42):
        super().__init__(env_config, seed)
        self.Na = 7
        self.action_space = Discrete(self.Na)

    def micro_policy(self, action):
        mg = self.mg
        load = mg.load
        pv = mg.pv
        capa_to_charge = mg.battery.capa_to_charge
        capa_to_discharge = mg.battery.capa_to_discharge
        policies = {
            "sell_excess": {
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": max(0, pv - load),
                "pv": min(pv, load),
                "genset": 0,
            },
            "store_excess": {
                "battery_charge": max(0, pv - load),
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "fill_battery_from_grid": {
                "battery_charge": capa_to_charge,
                "battery_discharge": 0,
                "grid_import": capa_to_charge + (load - pv),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "buy_for_load": {
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": max(0, load - pv),
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": 0,
            },
            "genset_for_load": {
                "battery_charge": 0,
                "battery_discharge": 0,
                "grid_import": 0,
                "grid_export": 0,
                "pv": min(pv, load),
                "genset": max(0, load - pv),
            },
            "discharge_to_sell": {
                "battery_charge": 0,
                "battery_discharge": capa_to_discharge,
                "grid_import": max(0, load - pv - capa_to_discharge),
                "grid_export": max(0, capa_to_discharge + pv - load),
                "pv": pv,
                "genset": 0,
            },
            "discharge_for_load": {
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
        return self.micro_policy(action)

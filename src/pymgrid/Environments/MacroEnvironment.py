import pymgrid.Environments.Environment as pymgridEnvs
import numpy as np
import operator

import gym

from copy import deepcopy

from ..algos.Control import RuleBasedControl as RBC


class MicroPolicy:
    """
    MicroPolicy Base Class
    """

    def __init__(self):
        pass

    def getAction(self, env):
        pass


class CurrentStatePolicy(MicroPolicy):
    """
    Policy which uses (only!) the current state of mg,
    processes it with policyModel(torch) and gives the action
    """

    def __init__(self, policyModel):
        """
        Agent policyModel -- object implementing method select_action mapping state |--> action
        """
        self.policyModel = policyModel

    def actionMap(self, agentAction, env):
        """
        Maps agent output to control_dict supported by pymgrid
        """
        # CSPLA!!!!
        return env.get_action_priority_list(agentAction)

    def getAction(self, env):
        """
        Based on the current state of mg returns action (in the form of control dict)
        Environment env -- microgrid environment with all the info
        """
        agentAction = self.policyModel.select_action(env.state)
        return self.actionMap(agentAction, env)


class RBCPolicy(MicroPolicy):
    def __init__(self, mg):
        self.microgrid = mg

    def getAction(self, env):

        # pymgrid log inits
        baseline_priority_list_update_status = deepcopy(env.mg._df_record_state)

        t = env.round
        if t is None:
            t = 0

        if env.mg.architecture["grid"] == 1:
            priorityDict = self._generate_priority_list(
                env.mg.architecture,
                env.mg.parameters,
                env.mg._grid_status_ts.iloc[t].values[0],
                env.mg._grid_price_import.iloc[t].values[0],
                env.mg._grid_price_export.iloc[t].values[0],
            )
        else:
            priorityDict = self._generate_priority_list(
                env.mg.architecture, env.mg.parameters
            )

        control_dict = self._run_priority_based(
            env.mg._load_ts.iloc[t].values[0],
            env.mg._pv_ts.iloc[t].values[0],
            env.mg.parameters,
            baseline_priority_list_update_status,
            priorityDict,
        )

        # pymgrid logging....

        # /pymgrid logging

        return control_dict

    def _run_priority_based(self, load, pv, parameters, status, priority_dict):
        """
        PYMGRID.CONTROL.RBC
        This function runs one loop of rule based control, based on a priority list, load and pv, dispatch the
        generators

        Parameters
        ----------
        load: float
            Demand value
        PV: float
            PV generation
        parameters: dataframe
            The fixed parameters of the mircrogrid
        status: dataframe
            The parameters of the microgrid changing with time.
        priority_dict: dictionnary
            Dictionnary representing the priority with which run each generator.

        """

        temp_load = load
        # todo add reserves to pymgrid
        excess_gen = 0

        p_charge = 0
        p_discharge = 0
        p_import = 0
        p_export = 0
        p_genset = 0
        load_not_matched = 0
        pv_not_curtailed = 0
        self_consumed_pv = 0

        sorted_priority = priority_dict
        min_load = 0
        if self.microgrid.architecture["genset"] == 1:
            # load - pv - min(capa_to_discharge, p_discharge) > 0: then genset on and min load, else genset off
            grid_first = 0
            capa_to_discharge = max(
                min(
                    (
                        status["battery_soc"][-1]
                        * parameters["battery_capacity"].values[0]
                        - parameters["battery_soc_min"].values[0]
                        * parameters["battery_capacity"].values[0]
                    )
                    * parameters["battery_efficiency"].values[0],
                    self.microgrid.battery.p_discharge_max,
                ),
                0,
            )

            if (
                self.microgrid.architecture["grid"] == 1
                and sorted_priority["grid"] < sorted_priority["genset"]
                and sorted_priority["grid"] > 0
            ):
                grid_first = 1

            if temp_load > pv + capa_to_discharge and grid_first == 0:

                min_load = (
                    self.microgrid.parameters["genset_rated_power"].values[0]
                    * self.microgrid.parameters["genset_pmin"].values[0]
                )
                if min_load <= temp_load:
                    temp_load = temp_load - min_load
                else:
                    temp_load = min_load
                    priority_dict = {"PV": 0, "battery": 0, "grid": 0, "genset": 1}

        sorted_priority = sorted(priority_dict.items(), key=operator.itemgetter(1))
        # for gen with prio i in 1:max(priority_dict)
        # we sort the priority list
        # probably we should force the PV to be number one, the min_power should be absorbed by genset, grid?
        # print (sorted_priority)
        for gen, priority in sorted_priority:  # .iteritems():

            if priority > 0:

                if gen == "PV":
                    self_consumed_pv = min(
                        temp_load, pv
                    )  # self.maximum_instantaneous_pv_penetration,
                    temp_load = max(0, temp_load - self_consumed_pv)
                    excess_gen = pv - self_consumed_pv
                    pv_not_curtailed = pv_not_curtailed + pv - excess_gen

                if gen == "battery":

                    capa_to_charge = max(
                        (
                            parameters["battery_soc_max"].values[0]
                            * parameters["battery_capacity"].values[0]
                            - status["battery_soc"][-1]
                            * parameters["battery_capacity"].values[0]
                        )
                        / self.microgrid.parameters["battery_efficiency"].values[0],
                        0,
                    )
                    capa_to_discharge = max(
                        (
                            status["battery_soc"][-1]
                            * parameters["battery_capacity"].values[0]
                            - parameters["battery_soc_min"].values[0]
                            * parameters["battery_capacity"].values[0]
                        )
                        * parameters["battery_efficiency"].values[0],
                        0,
                    )
                    if temp_load > 0:
                        p_discharge = max(
                            0,
                            min(
                                capa_to_discharge,
                                parameters["battery_power_discharge"].values[0],
                                temp_load,
                            ),
                        )
                        temp_load = temp_load - p_discharge

                    elif excess_gen > 0:
                        p_charge = max(
                            0,
                            min(
                                capa_to_charge,
                                parameters["battery_power_charge"].values[0],
                                excess_gen,
                            ),
                        )
                        excess_gen = excess_gen - p_charge

                        pv_not_curtailed = pv_not_curtailed + p_charge

                if gen == "grid":
                    if temp_load > 0:
                        p_import = temp_load
                        temp_load = 0

                    elif excess_gen > 0:
                        p_export = excess_gen
                        excess_gen = 0

                        pv_not_curtailed = pv_not_curtailed + p_export

                if gen == "genset":
                    if temp_load > 0:
                        p_genset = temp_load + min_load
                        temp_load = 0
                        min_load = 0

        if temp_load > 0:
            load_not_matched = 1

        control_dict = {
            "battery_charge": p_charge,
            "battery_discharge": p_discharge,
            "genset": p_genset,
            "grid_import": p_import,
            "grid_export": p_export,
            "loss_load": load_not_matched,
            "pv_consummed": pv_not_curtailed,
            "pv_curtailed": pv - pv_not_curtailed,
            "load": load,
            "pv": pv,
        }

        return control_dict

    def _generate_priority_list(
        self, architecture, parameters, grid_status=0, price_import=0, price_export=0
    ):
        """
        PYMGRID.CONTROL.RBC
        Depending on the architecture of the microgrid and grid related import/export costs, this function generates a
        priority list to be run in the rule based benchmark.
        """
        # compute marginal cost of each resource
        # construct priority list
        # should receive fuel cost and cost curve, price of electricity
        if architecture["grid"] == 1:

            if (
                price_export / (parameters["battery_efficiency"].values[0] ** 2)
                < price_import
            ):

                # should return something like ['gen', starting at in MW]?
                priority_dict = {
                    "PV": 1 * architecture["PV"],
                    "battery": 2 * architecture["battery"],
                    "grid": int(3 * architecture["grid"] * grid_status),
                    "genset": 4 * architecture["genset"],
                }

            else:
                # should return something like ['gen', starting at in MW]?
                priority_dict = {
                    "PV": 1 * architecture["PV"],
                    "battery": 3 * architecture["battery"],
                    "grid": int(2 * architecture["grid"] * grid_status),
                    "genset": 4 * architecture["genset"],
                }

        else:
            priority_dict = {
                "PV": 1 * architecture["PV"],
                "battery": 2 * architecture["battery"],
                "grid": 0,
                "genset": 4 * architecture["genset"],
            }

        return priority_dict


############ NOT IMPLEMENTED ####################
# class HistoryStatePolicy(MicroPolicy):
#     '''
#         Policy which uses the current state of mg and history of states,
#         processes it with policyModel and gives the action
#     '''

#     def __init__(self,policyModel):
#         '''
#         TorchModel policyModel
#         '''
#         self.policyModel=policyModel

#     def actionMap(self, torchAction):
#         '''
#         Maps torch output to control_dict supported by pymgrid
#         '''
#         #CSPLA!!!!
#         return self.get_action_priority_list(torchAction)

#     #NOTIMPLEMENTED!!!!
#     def getAction(self, env):
#         '''
#         Based on the current state of mg returns action (in the form of control dict)
#         Environment env -- microgrid environment with all the info
#         '''

#         torchAction = self.policyModel(env.state) #???????????????????????????
#         return self.actionMap(torchAction)


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

        self.switchingFrequency = switchingFrequency
        self.microPolicies = microPolicies

        # microPolicy action design
        self.Na = len(self.microPolicies)

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
            control_dict = self.microPolicies[action].getAction(self)
            # print("CD:", control_dict)

            self.mg.run(control_dict)

            # COMPUTE NEW STATE AND REWARD
            self.state = self.transition()
            self.reward += self.get_reward()
            self.done = self.mg.done
            self.info = {}
            self.round += 1

        return self.state, self.reward, self.done, self.info

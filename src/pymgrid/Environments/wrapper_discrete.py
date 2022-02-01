def get_action_discrete(self, action):
    """
    :param action: current action
    :return: control_dict : dicco of controls
    """
    """
        Actions are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        """
    control_dict = {}

    control_dict["pv_consumed"] = action[0]
    scaler_battery
    if self.mg.architecture["battery"] == 1:
        if action[1]>=0:
            control_dict["battery_charge"] = action[1] * scaler_battery
        else:
            control_dict["battery_discharge"] = -action[1] * scaler_battery
        control_dict["battery_charge"] = action[1] * action[3]
        control_dict["battery_discharge"] = action[2] * (1 - action[3])

    if self.mg.architecture["genset"] == 1:
        control_dict["genset"] = action[4]

        if self.mg.architecture["grid"] == 1:
            control_dict["grid_import"] = action[5] * action[7]
            control_dict["grid_export"] = action[6] * (1 - action[7])

    elif self.mg.architecture["grid"] == 1:
        control_dict["grid_import"] = action[4] * action[6]
        control_dict["grid_export"] = action[5] * (1 - action[6])

    return control_dict

from typing import Union
import numpy as np
import turbine_functions as F
from gas_model import (
    get_s_from_state,
    get_h_from_state,
    get_sound_speed_from_state,
)


default_types = Union[int, float, np.ndarray]


class TurbineStage:
    u_div_c: default_types
    blade_efficiency: default_types
    blade_power: default_types

    def __init__(
            self,
            mass_flow_rate: default_types,
            average_diameter: default_types,
            inlet_pressure: default_types,
            inlet_temperature: default_types,
            outlet_pressure: default_types,
            rotation_speed: default_types,
            degree_of_reaction: default_types,
            stator_outlet_angle: default_types,
            overlapping: default_types,
            inlet_speed: default_types = 0,
            is_last: bool = True,
    ) -> None:
        self.mass_flow_rate = mass_flow_rate
        self.average_diameter = average_diameter
        self.inlet_pressure = inlet_pressure
        self.inlet_temperature = inlet_temperature
        self.outlet_pressure = outlet_pressure
        self.rotation_speed = rotation_speed
        self.degree_of_reaction = degree_of_reaction
        self.stator_outlet_angle = stator_outlet_angle
        self.overlapping = overlapping
        self.inlet_speed = inlet_speed
        self.is_last = is_last

        self.blade_efficiency, self.u_div_c, self.blade_power = self.design()

    def design(self):
        chord = 50 / 1000

        u = np.pi * self.rotation_speed * self.average_diameter
        total_heat_drop, stator_heat_drop, rotor_heat_drop = F.compute_heat_drops(
            p0=self.inlet_pressure,
            t0=self.inlet_temperature,
            p2=self.outlet_pressure,
            dor=self.degree_of_reaction,
            inlet_speed=self.inlet_speed
        )
        dummy_speed = (2 * total_heat_drop) ** 0.5
        u_div_c = u / dummy_speed

        h0 = get_h_from_state(p=self.inlet_pressure, t=self.inlet_temperature)
        s0 = get_s_from_state(p=self.inlet_pressure, h=h0)
        p1, v1t, t1t, h1t = F.compute_intermedia_point(h0, s0, stator_heat_drop)

        c1t = (2 * stator_heat_drop) ** 0.5
        a = get_sound_speed_from_state(h1t, s0)
        mach_1t = c1t / a
        if (mach_1t > 1).any():
            raise RuntimeError("M1t > 1")

        l1 = F.compute_stator_blade_length(
            self.mass_flow_rate, self.average_diameter, c1t, v1t, self.stator_outlet_angle, chord
        )
        fi = F.compute_speed_coefficient(blade_length=l1, chord=chord, is_rotor=False)
        nu1 = F.compute_discharge_coefficient(blade_length=l1, chord=chord, is_rotor=False)
        alpha_1 = F.move_angle(self.stator_outlet_angle, fi, nu1)

        c1 = fi * c1t
        w1, beta_1 = F.compute_triangle(u=u, inlet_speed=c1, inlet_angle=alpha_1)
        stator_loss = (c1t ** 2 - c1 ** 2) / 2

        h1 = h1t + stator_loss
        s1 = get_s_from_state(p=p1, h=h1)

        l2 = l1 + self.overlapping
        psi = F.compute_speed_coefficient(blade_length=l2, chord=chord, is_rotor=True)
        nu2 = F.compute_discharge_coefficient(blade_length=l2, chord=chord, is_rotor=True)
        _, v2t, t2t, h2t = F.compute_intermedia_point(h1, s1, rotor_heat_drop)
        w2t = (2 * rotor_heat_drop + w1 ** 2) ** 0.5

        a = get_sound_speed_from_state(h2t, s1)
        mach_2t = w2t / a
        if (mach_2t > 1).any():
            raise RuntimeError("M2t > 1")

        w2 = psi * w2t
        beta_2_eff = F.compute_beta_2(
            mass_flow_rate=self.mass_flow_rate, d=self.average_diameter, w2t=w2t, v2t=v2t, l2=l2, nu=nu2
        )
        beta_2 = F.move_angle(beta_2_eff, psi, nu2)

        c2, alpha_2 = F.compute_triangle(u=u, inlet_speed=w2, inlet_angle=beta_2)

        output_speed_loss = (c2 ** 2) / 2

        work_pu = u * (c1 * np.cos(alpha_1) + c2 * np.cos(alpha_2))
        available_energy = total_heat_drop - int(1 - self.is_last) * output_speed_loss
        blade_efficiency = work_pu / available_energy
        power = work_pu * self.mass_flow_rate

        return blade_efficiency, u_div_c, power

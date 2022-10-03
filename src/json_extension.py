from enum import Enum
import json
import inspect
import sys

import numpy as np


from AaronTools.json_extension import ATEncoder, ATDecoder
from milo.containers import (
    Positions,
    Velocities,
    Accelerations,
    Forces,
    Frequencies,
    ForceConstants,
    Masses,
    Time,
    Energies,
)
from milo import enumerations as enums
from milo.electronic_structure_program_handler import ProgramHandler
from milo.random_number_generator import RandomNumberGenerator


class MiloEncoder(ATEncoder):
    def default(self, obj):
        # encode numpy arrays
        if isinstance(obj, ProgramHandler):
            return None

        if isinstance(obj, np.ndarray):
            return self._encode_ndarray(obj)
        
        if isinstance(obj, Enum):
            return self._encode_enum(obj)

        # check to see if we can encode this
        # the encode function should be named "_encode_" + the lower-case
        # name of the class
        encode_name = "_encode_%s" % obj.__class__.__name__.casefold()
        try:
            encode_func = getattr(self, encode_name)
            return encode_func(obj)
        
        except AttributeError:
            # fall back to default JSON encoding methods
            return super().default(obj)
    
    def _encode_enum(self, obj):
        data = {
            "_type": "Enum",
            "specific_type": obj.__class__.__name__,
            "value": obj.value,
        }
        return data

    def _encode_accelerations(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_meter_per_sec_sqrd()
        }
        return data
    
    def _encode_energies(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_hartree()
        }
        return data
    
    def _encode_forceconstants(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_newton_per_meter()
        }
        return data

    def _encode_forces(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_newton()
        }
        return data
    
    def _encode_frequencies(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_recip_cm()
        }
        return data
    
    def _encode_masses(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_amu()
        }
        return data
    
    def _encode_ndarray(self, obj):
        data = {
            "_type": "ndarray",
            "values": obj.real.tolist(),
            "imag": False,
            "dtype": repr(obj.dtype),
        }
        if np.iscomplexobj(obj):
            data["imag"] = obj.imag   
        return data
    
    def _encode_positions(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_angstrom(),
        }
        return data

    def _encode_randomnumbergenerator(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "seed": obj.seed,
            "state": obj.rng.getstate(),
        }
        return data

    def _encode_time(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_femtosecond(),
        }
        return data

    def _encode_velocities(self, obj):
        data = {
            "_type": obj.__class__.__name__,
            "values": obj.as_meter_per_sec(),
        }
        return data
    


class MiloDecoder(ATDecoder):

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        
        decode_name = "_decode_%s" % obj["_type"].casefold()
        try:
            decode_func = getattr(self, decode_name)
            return decode_func(obj)
        except AttributeError:
            return super().object_hook(obj)

    def _decode_accelerations(self, obj):
        out = Accelerations()
        for entry in obj["values"]:
            out.append(*entry, enums.AccelerationUnits.METER_PER_SEC_SQRD)
        return out
    
    def _decode_energies(self, obj):
        out = Energies()
        for entry in obj["values"]:
            out.append(entry, enums.EnergyUnits.HARTREE)
        return out
   
    def _decode_enum(self, obj):
        if "specific_type" not in obj:
            return None
        for obj_name, cls in inspect.getmembers(sys.modules["milo.enumerations"]):
            if obj_name == obj["specific_type"] and inspect.isclass(cls):
                out = cls(obj["value"])
                break

        return out

    def _decode_forceconstants(self, obj):
        out = ForceConstants()
        for entry in obj["values"]:
            out.append(entry, enums.ForceConstantUnits.NEWTON_PER_METER)
        return out
    
    def _decode_forces(self, obj):
        out = Forces()
        for entry in obj["values"]:
            out.append(*entry, enums.ForceUnits.NEWTON)
        return out

    def _decode_frequencies(self, obj):
        out = Frequencies()
        for entry in obj["values"]:
            out.append(entry, enums.FrequencyUnits.RECIP_CM)
        return out
    
    def _decode_masses(self, obj):
        out = Masses()
        for entry in obj["values"]:
            out.append(entry, enums.MassUnits.AMU)
        return out
    
    def _decode_ndarray(self, obj):
        out = np.array(
            obj["values"], dtype=eval("np.%s" % obj["dtype"]),
        )
        if obj["imag"] is not False:
            out.imag = obj["imag"]
        return out
    
    def _decode_positions(self, obj):
        out = Positions()
        for entry in obj["values"]:
            out.append(*entry, enums.DistanceUnits.ANGSTROM)
        return out
   
    def _decode_randomnumbergenerator(self, obj):
        out = RandomNumberGenerator(obj["seed"])
        state = []
        for item in obj["state"]:
            if isinstance(item, list):
                state.append(tuple(item))
            else:
                state.append(item)

        out.rng.setstate(tuple(state))
        return out

    def _decode_time(self, obj):
        out = Time(obj["values"], enums.TimeUnits.FEMTOSECOND)
        return out
    
    def _decode_velocities(self, obj):
        out = Velocities()
        for entry in obj["values"]:
            out.append(*entry, enums.VelocityUnits.METER_PER_SEC)
        return out
    

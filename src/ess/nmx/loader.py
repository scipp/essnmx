# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import scipp as sc
import scippnexus as snx

from .logging import NMXLogger

InputFileName = NewType("InputFileName", str)
Events = NewType("Events", sc.DataArray)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100000)


class McStasEventDataSchema(str):
    """McStas event data schema, suffix of the event data path."""

    ...


DefaultMcStasEventDataSchema = McStasEventDataSchema("p_x_y_n_id_t")
McStasEventDataSchemaWithWaveLength = McStasEventDataSchema("p_x_y_n_id_t_L_L")


def _get_entry_path_mcstas(data_path_suffix: McStasEventDataSchema) -> str:
    if data_path_suffix is None:
        data_path_suffix = DefaultMcStasEventDataSchema

    return f"entry1/data/bank01_events_dat_list_{data_path_suffix}"


def _copy_partial_var(
    var: sc.Variable, dim: str, idx: int, unit: str, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve property from variable."""
    original_var = var[dim, idx]
    var = original_var.copy().astype(dtype) if dtype else original_var.copy()
    var.unit = sc.Unit(unit)
    return var


def _read_mcstas_nmx_file(
    *, file: snx.File, entry_path: str, max_prop: MaximumProbability
) -> sc.DataGroup:
    var: sc.Variable = file[entry_path]["events"][()].rename_dims(
        {'dim_0': 'event', 'dim_1': 'property'}
    )

    property_recipes = {
        'weights': {"idx": 0, "unit": 'counts'},
        'x_list': {"idx": 1, "unit": 'm'},
        'y_list': {"idx": 2, "unit": 'm'},
        'id_list': {"idx": 4, "unit": 'dimensionless', "dtype": "int64"},
        't_list': {"idx": 5, "unit": 's'},
    }
    dg = sc.DataGroup(
        **{
            name: _copy_partial_var(var, dim='property', **recipe)
            for name, recipe in property_recipes.items()
        }
    )

    weights: sc.Variable = dg.pop('weights')
    dg['weights'] = (max_prop / weights.max()) * weights

    return dg


def read_file(
    file_name: InputFileName,
    mcstas_data_schema: Optional[McStasEventDataSchema] = None,
    max_prop: Optional[MaximumProbability] = None,
    logger: Optional[NMXLogger] = None,
) -> Events:
    with snx.File(file_name) as file:
        if "entry1" in file:
            suffix = mcstas_data_schema or DefaultMcStasEventDataSchema
            prop = max_prop or DefaultMaximumProbability
            dg = _read_mcstas_nmx_file(
                file=file, entry_path=_get_entry_path_mcstas(suffix), max_prop=prop
            )
        elif 'entry' in file:
            raise NotImplementedError("Measurement data loader is not implemented yet.")
        else:
            raise ValueError(f"Can not load {file_name} with NMX file loader.")

    weights: sc.Variable = dg['weights']
    t_list: sc.Variable = dg['t_list']
    id_list: sc.Variable = dg['id_list']
    x_list: sc.Variable = dg['x_list']
    y_list: sc.Variable = dg['y_list']

    if logger:
        logger.info(f"Read {len(weights)} events from {file_name}.")
        logger.info(f"t-list range: ({t_list.min().value}, {t_list.max().value}).")

    return Events(
        sc.DataArray(
            data=weights, coords={'t': t_list, 'id': id_list, 'x': x_list, 'y': y_list}
        )
    )

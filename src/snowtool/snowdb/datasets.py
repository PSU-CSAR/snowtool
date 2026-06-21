"""The SNODAS dataset definition and the built-in dataset registry.

The source of truth for the SNODAS dataset: its variables and its grid/DEM spec
(the geometry/range literals live here, on ``SNODAS_SPEC``). ``Product`` is the
SNODAS variable enum, including its filename-code mapping (``to_glob`` /
``from_product_codes``); the *filename parser* that consumes it lives with the
ingest in :mod:`snowtool.snowdb.input_rasters` — the only place that parses
filenames (the read path is dataset-agnostic). ``DEFAULT_DATASET_SPECS`` is what
the app/CLI pass to a ``SnowDb``. When a second dataset is added, split a
per-dataset module out of here.
"""

from enum import StrEnum
from typing import Self

from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

# --- SNODAS products: the variables + their filename-code mapping --------------

_product_code_to_product_name = {
    1025: 'precip',
    1034: 'swe',
    1036: 'depth',
    1038: 'average_temp',
    1039: 'sublimation_blowing',
    1044: 'runoff',
    1050: 'sublimation',
}

_product_name_to_product_code = {v: k for k, v in _product_code_to_product_name.items()}


_millimeters = Unit(name='mm', scale_factor=1)
_millimeters_100 = Unit(name='mm', scale_factor=100)
_kelvin = Unit(name='k', scale_factor=1)
_kg_per_meter2 = Unit(name='kg_per_m2', scale_factor=10)

_units = {
    'precip_solid': _kg_per_meter2,
    'precip_liquid': _kg_per_meter2,
    'swe': _millimeters,
    'depth': _millimeters,
    'average_temp': _kelvin,
    'sublimation': _millimeters_100,
    'sublimation_blowing': _millimeters_100,
    'runoff': _millimeters_100,
}


class Product(StrEnum):
    PRECIP_SOLID = 'precip_solid'
    PRECIP_LIQUID = 'precip_liquid'
    SNOW_WATER_EQUIVALENT = 'swe'
    SNOW_DEPTH = 'depth'
    AVERAGE_TEMP = 'average_temp'
    SUBLIMATION = 'sublimation'
    SUBLIMATION_BLOWING = 'sublimation_blowing'
    RUNOFF = 'runoff'

    @classmethod
    def from_product_codes(cls: type[Self], product_code: int, vcode: str) -> Self:
        product_name: str = _product_code_to_product_name[product_code]

        if product_name != 'precip':
            return cls(product_name)

        match vcode:
            case 'lL00':
                return cls('precip_liquid')
            case 'lL01':
                return cls('precip_solid')
            case _:
                raise ValueError(
                    f"unknown vcode '{vcode}' for product type 'precip'",
                )

    def to_glob(self: Self) -> str:
        product_name = self.value
        vcode: str = ''

        if product_name.startswith('precip'):
            product_name, precip_type = product_name.split('_')
            match precip_type:
                case 'liquid':
                    vcode = 'lL00'
                case 'solid':
                    vcode = 'lL01'
                case _:
                    raise ValueError(f"unknown precip type '{precip_type}'")

        product_code = _product_name_to_product_code[product_name]
        return ''.join(
            [
                '?' * 6,
                'v[01]',
                str(product_code),
                '?',
                vcode,
                '*TTNATS*.tif',
            ],
        )

    def unit(self: Self) -> Unit:
        return _units[self.value]


# --- SNODAS variables + spec (the source of truth for SNODAS values) ----------

# SNODAS variables, one per product. All are intensive quantities (depths,
# temperature) reported as area-weighted means; reads are int16 with the SNODAS
# nodata sentinel. (Switching any to an INTEGRAL total is a future, domain-driven
# change -- see the plan's reduced-unit note.)
SNODAS_VARIABLES = tuple(
    DatasetVariable(
        key=product.value,
        unit=product.unit(),
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=-9999.0,
        glob=product.to_glob(),
    )
    for product in Product
)

SNODAS_SPEC = DatasetSpec(
    name='snodas',
    grid_params=GridParams(
        origin_x=-124.733333333333333,
        origin_y=52.875000000000000,
        px_size=0.008333333333333,
        cols=6935,
        rows=3351,
        tile_size=256,
    ),
    # Overall min/max DEM elevation (m); bounds the elevation bands.
    dem_min_m=-84.833877563477,
    dem_max_m=4291.7211914062,
    variables=SNODAS_VARIABLES,
)


# --- registry -----------------------------------------------------------------

# The built-in datasets; the app/CLI pass this to SnowDb. Tests may pass a subset
# or their own synthetic specs.
DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (SNODAS_SPEC,)

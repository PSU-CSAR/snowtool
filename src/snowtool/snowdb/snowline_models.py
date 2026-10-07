from pydantic import BaseModel, Field


class SnowLine(BaseModel):
    elevation_ft: float = Field(
        description='Interpolated elevation where the variable crosses the threshold.',
    )
    lower_band_area_m2: float = Field(
        description='Lower bounding area below the crossing; lower values mean'
        'that the interpolation rests on a few pixels',
    )
    higher_band_area_m2: float = Field(
        description='Higher bounding area above the crossing',
    )
    interpolation_span_ft: float = Field(
        description='Distance between the two band centers. Larger span'
        'means a coarser result',
    )

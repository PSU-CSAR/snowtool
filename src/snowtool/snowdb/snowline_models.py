from pydantic import BaseModel


class SnowLine(BaseModel):
    elevation_ft: float
    lower_band_area_m2: float
    higher_band_area_m2: float
    interpolation_span_ft: float

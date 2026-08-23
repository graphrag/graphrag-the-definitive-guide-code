from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple
from pydantic import BaseModel


# --- Core entities -----------------------------------------------------------

class Airport(BaseModel):
    airport_code: str
    city: str
    # Simple representation of geopoint as (lat, lon).
    # You can swap this for a more sophisticated type if needed.
    location: Tuple[float, float]


class Aircraft(BaseModel):
    aircraft_id: str
    model_number: str
    registration_date: datetime

    # OPERATES_AT: MANY_TO_ONE from Aircraft -> Airport
    operates_at: Optional[Airport] = None


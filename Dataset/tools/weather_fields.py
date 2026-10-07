"""Accepted weather value keys and source aliases for L0 and render-ready rows."""

WEATHER_NUMERIC_FIELDS = frozenset({
    "rain", "wetness", "fog_density", "dust", "wind_speed",
    "wind_direction_deg", "visibility_m", "temperature_c",
    "illumination_lux", "hazard_concentration_ppm", "hazard_radius_m",
})
WEATHER_BOOLEAN_FIELDS = frozenset({"hazard_source_active"})
WEATHER_ALIASES = {"fog": "fog_density", "visibility": "visibility_m"}
WEATHER_FRACTION_FIELDS = frozenset({"rain", "fog_density"})
WEATHER_OVERRIDE_FIELDS = (
    WEATHER_NUMERIC_FIELDS | WEATHER_BOOLEAN_FIELDS | WEATHER_ALIASES.keys()
)
WEATHER_SOURCE_FIELDS = WEATHER_OVERRIDE_FIELDS | {
    "tick", "condition", "temperature_source"
}

"""ITU-R P.838-3 uniform-path rain attenuation, with explicit mm/h inputs.

Equations 1--5 and Tables 1--4, official 2005-03 revision:
https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf
This is propagation absorption, not a packet-loss multiplier, wet-antenna
model, cloud/fog loss, or a local weather calibration. Uniform rain over the
whole Tx--Rx segment is an authored assumption, not inferred from weather.rain.
"""
from __future__ import annotations

import math

SOURCE = "ITU-R P.838-3 (03/2005), equations 1-5, Tables 1-4"
COEFFICIENTS = {
    "k_h": ((-5.33980, -.35351, -.23789, -.94158),
            (-.10008, 1.26970, .86036, .64552),
            (1.13098, .45400, .15354, .16817), -.18961, .71147),
    "k_v": ((-3.80595, -3.44965, -.39902, .50167),
            (.56934, -.22911, .73042, 1.07319),
            (.81061, .51059, .11899, .27195), -.16398, .63297),
    "alpha_h": ((-.14318, .29591, .32177, -5.37610, 16.1721),
                (1.82442, .77564, .63773, -.96230, -3.29980),
                (-.55187, .19822, .13164, 1.47828, 3.43990), .67849, -1.95537),
    "alpha_v": ((-.07771, .56727, -.20238, -48.2991, 48.5833),
                (2.33840, .95545, 1.14520, .791669, .791459),
                (-.76284, .54039, .26809, .116226, .116479), -.053739, .83433),
}


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def frequency_coefficients(frequency_ghz):
    f = _number(frequency_ghz, "frequency_ghz")
    if not 1 <= f <= 1000:
        raise ValueError("ITU-R P.838-3 frequency must be in [1,1000] GHz")
    x = math.log10(f)
    result = {}
    for name, (a, b, c, m, offset) in COEFFICIENTS.items():
        value = sum(ai * math.exp(-((x - bi) / ci) ** 2)
                    for ai, bi, ci in zip(a, b, c)) + m * x + offset
        result[name] = 10 ** value if name.startswith("k_") else value
    return result


def specific_attenuation_db_per_km(frequency_ghz, rain_rate_mm_per_h,
                                 elevation_deg, polarization_tilt_deg):
    rain = _number(rain_rate_mm_per_h, "rain_rate_mm_per_h")
    theta = _number(elevation_deg, "elevation_deg")
    tilt = _number(polarization_tilt_deg, "polarization_tilt_deg")
    if rain < 0 or not -90 <= theta <= 90 or not 0 <= tilt <= 90:
        raise ValueError("rain must be nonnegative; elevation [-90,90], tilt [0,90] degrees")
    c = frequency_coefficients(frequency_ghz)
    factor = math.cos(math.radians(theta)) ** 2 * math.cos(2 * math.radians(tilt))
    k = (c['k_h'] + c['k_v'] + (c['k_h'] - c['k_v']) * factor) / 2
    alpha = (c['k_h'] * c['alpha_h'] + c['k_v'] * c['alpha_v']
             + (c['k_h'] * c['alpha_h'] - c['k_v'] * c['alpha_v']) * factor) / (2 * k)
    return 0.0 if rain == 0 else k * rain ** alpha


def uniform_path_attenuation_db(frequency_ghz, rain_rate_mm_per_h,
                                tx_enu_m, rx_enu_m, polarization_tilt_deg):
    if len(tx_enu_m) != 3 or len(rx_enu_m) != 3:
        raise ValueError("Tx/Rx require three ENU metre coordinates")
    delta = [_number(b, 'rx_enu_m') - _number(a, 'tx_enu_m')
             for a, b in zip(tx_enu_m, rx_enu_m)]
    distance = math.hypot(*delta)
    theta = 0 if distance == 0 else math.degrees(math.asin(delta[2] / distance))
    return specific_attenuation_db_per_km(frequency_ghz, rain_rate_mm_per_h,
                                         theta, polarization_tilt_deg) * distance / 1000


def resolve_rain_profile(document, duration_ns):
    required = {'schema_version', 'source_ref', 'assumption', 'polarization_tilt_deg', 'windows'}
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("Rain profile requires the exact declared fields")
    if document['schema_version'] != 'p09.rain.itu838.uniform-path/v1':
        raise ValueError("Unsupported rain profile version")
    for key in ('source_ref', 'assumption'):
        if not isinstance(document[key], str) or not document[key]:
            raise ValueError(f"Rain {key} must declare its authored provenance")
    tilt = _number(document['polarization_tilt_deg'], 'polarization_tilt_deg')
    if not 0 <= tilt <= 90:
        raise ValueError("Polarization tilt must be in [0,90] degrees")
    if not isinstance(document['windows'], list) or not document['windows']:
        raise ValueError("Rain profile requires explicit windows; omitted times have zero rain")
    last = -1
    for row in document['windows']:
        if not isinstance(row, dict) or set(row) != {'start_ns', 'end_ns', 'rain_rate_mm_per_h', 'unit'}:
            raise ValueError("Rain window requires start_ns/end_ns/rain_rate_mm_per_h/unit")
        if (type(row['start_ns']) is not int or type(row['end_ns']) is not int
                or not 0 <= row['start_ns'] < row['end_ns'] <= duration_ns or row['start_ns'] < last):
            raise ValueError("Rain windows must be sorted, nonoverlapping, inside the episode")
        if row['unit'] != 'mm/h' or _number(row['rain_rate_mm_per_h'], 'rain_rate_mm_per_h') < 0:
            raise ValueError("Rain rate must explicitly use nonnegative mm/h, never a ratio")
        last = row['end_ns']
    return document

# unit choices. Factors convert to metres, days and cubic metres

LENGTH = {'m': 1.0, 'km': 1000.0, 'ft': 0.3048}
CONDUCTIVITY = {'m/d': 1.0, 'm/s': 86400.0, 'ft/d': 0.3048}
TRANSMISSIVITY = {'m²/d': 1.0, 'm²/s': 86400.0, 'ft²/d': 0.09290304}
FLOW = {'L/s': 86.4, 'ML/d': 1000.0, 'm³/d': 1.0, 'ML/yr': 1000.0 / 365.25,
        'GL/yr': 1.0e6 / 365.25, 'US gal/min': 5.450992992}
VOLUME = {'ML': 1000.0, 'm³': 1.0, 'GL': 1.0e6}
DRAWDOWN = {'m': 1.0, 'cm': 0.01, 'ft': 0.3048}

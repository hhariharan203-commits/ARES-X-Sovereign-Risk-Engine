def generate_intelligence(row, score):

    drivers = []

    if row["inflation"] > 6:
        drivers.append("Inflation pressure building")

    if row["gdp_growth"] < 1:
        drivers.append("Growth slowdown detected")

    if row["unemployment"] > 8:
        drivers.append("Labor market weakening")

    if score > 0.75:
        regime = "CRISIS RISK REGIME"
        action = "Reduce exposure, shift to safe assets, hedge currency risk"
    elif score > 0.5:
        regime = "EARLY WARNING REGIME"
        action = "Rebalance portfolio, monitor macro signals"
    else:
        regime = "STABLE REGIME"
        action = "Maintain allocation"

    return {
        "regime": regime,
        "drivers": drivers,
        "action": action
    }

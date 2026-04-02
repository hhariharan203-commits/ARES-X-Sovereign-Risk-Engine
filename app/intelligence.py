def generate_brief(row, risk):

    # Situation
    if risk > 0.8:
        situation = "Severe macroeconomic instability"
    elif risk > 0.5:
        situation = "Emerging macro stress"
    else:
        situation = "Stable macro environment"

    # Diagnosis
    drivers = []

    if row["inflation"] > 6:
        drivers.append("Inflation pressure rising")

    if row["gdp_growth"] < 1:
        drivers.append("Growth slowdown")

    if row["unemployment"] > 8:
        drivers.append("Labor market weakness")

    if row["interest_rate"] > 7:
        drivers.append("Tight monetary conditions")

    if not drivers:
        drivers.append("No major imbalances")

    # Impact
    if risk > 0.8:
        impact = "High probability of capital flight and currency depreciation"
    elif risk > 0.5:
        impact = "Increased volatility across equity and FX markets"
    else:
        impact = "Stable conditions with low systemic risk"

    # Decision
    if risk > 0.8:
        decision = "Exit or hedge exposure immediately"
    elif risk > 0.5:
        decision = "Reduce exposure and rebalance portfolio"
    else:
        decision = "Maintain or increase exposure"

    return {
        "situation": situation,
        "drivers": drivers,
        "impact": impact,
        "decision": decision
    }

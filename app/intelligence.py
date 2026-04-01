def insight(score):
    if score > 0.75:
        return "🔴 High systemic risk driven by macro instability."
    elif score > 0.5:
        return "🟠 Rising risk — early warning signals emerging."
    else:
        return "🟢 Stable macroeconomic environment."

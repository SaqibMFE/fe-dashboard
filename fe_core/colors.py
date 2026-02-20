TEAM_MAP = {
    "Porsche":  ["WEH","MUE"],
    "Jaguar":   ["EVA","DAC"],
    "Nissan":   ["ROW","NAT"],
    "Mahindra": ["DEV","MOR"],
    "DS":       ["BAR","GUE"],
    "Andretti": ["DEN","DRU"],
    "Citroen":  ["CAS","JEV"],
    "Envision": ["BUE","ERI"],
    "Kiro":     ["TIC","MAR"],
    "Lola":     ["DIG","MAL"]
}

TEAM_COLOUR = {
    "Porsche":  "#6A0DAD",
    "Jaguar":   "#808080",
    "Nissan":   "#FF69B4",
    "Mahindra": "#D62728",
    "DS":       "#DAA520",
    "Andretti": "#1F77B4",
    "Citroen":  "#7DF9FF",
    "Envision": "#2CA02C",
    "Kiro":     "#8B4513",
    "Lola":     "#FFD700"
}

DRIVER_COLOUR = {d: TEAM_COLOUR[t] for t, drivers in TEAM_MAP.items() for d in drivers}

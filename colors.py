def PickColorTheme(idx):
    colors = [  ['#F1F1F2', '#BCBABE', '#A1D6E6', '#1995AD'],
                ['#FCE53D', '#FFA614', '#CA4026', '#7E1331'],
                ['#91D4C2', '#45BB89', '#3D82AB', '#003853'],
                ['#E5EDF8', '#E29E93', '#EDBC7A', '#0384BD'],
                ['#97BAA4', '#499360', '#295651', '#232941'],
                ['#D387D8', '#A13E97', '#632A7E', '#280E3B'],
                ['#8A54A2', '#8AD5EB', '#5954A4', '#04254E'],    ]

    idx %= len(colors)
    return colors[idx]
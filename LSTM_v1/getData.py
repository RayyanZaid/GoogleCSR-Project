# Get each Possession
    # 1. Each Possession ends on a terminal action OR
    # 2. Each Possesion ends when the shot clock increases


# Split each Possession into sequences of temporal windows

    # 1. Ensure that ALL the terminal actions are included in the temporal window
        # a) That means starting from a terminal action and going backwards
    # 2. Decide whether temporal windows should overlap or not


# Each temporal window has a sequence of moments


# Each moment should look like this :

# INPUT MATRIX : [ [  [m1], [m2], [m3], ... [m128]     ] ]  where m1 = [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]          

# LABEL VECTOR : [  [L1], [L2],         ... [L128]       ]
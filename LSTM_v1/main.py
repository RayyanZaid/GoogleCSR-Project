# Access the moments.csv file and extract the temporal windows 

# Temporal Window --> LSTM for training



# the last T=128 moments    +     r (buffer moments)

# t : the time of interest
# T = 128 moments (5 seconds) 
# r = 48 moments (1.5 seconds)

# temporal window = t - (T + r) --> t - r
# or : (t - r) - T --> t - r


# INPUT MATRIX : [ [  [m1], [m2], [m3], ... [m128]     ] ]                   

# LABEL VECTOR : [  [L1], [L2],         ... [L128]       ]


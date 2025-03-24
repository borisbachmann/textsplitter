# graph-chunking specific constants
DEFAULT_K = 5
DEFAULT_RESOLUTION = 1.0
DEFAULT_RANDOM_STATE = 0

# linear-chunking specific constants
DEFAULT_MAX_LENGTH = 512                # maximal length of produced chunks
DEFAULT_THRESHOLD = 0.3                 # similarity threshold for chunking
DEFAULT_METRIC = "pairwise"             # strategy to check threshold
DEFAULT_RES_MULTIPLIER = 2.875          # multiplier to calculate resolution
                                        # from goal length for graph chunking
                                        # (obtained by experimentation)
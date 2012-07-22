__author__ = 'denest'

import pstats, cProfile
import test_filters

cProfile.runctx("test_filters", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
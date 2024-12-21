# Copyright 2024 Carnegie Mellon University

from alhazen import IteratedExperiment
from itertools import repeat
import matplotlib.pyplot as plt
import pandas as pd
from pyibl import Agent
from tqdm import trange

LAG = 1
MISMATCH = 1
NOISE = 0.25
DECAY = 0.5
NONE_MATCH = 0.5
PREPOPULATED = 12

MOVES = ("D", "C")

PAYOFFS = (((-1,  -1), (10, -10)),
           ((-10, 10),  (1,   1)))

def move_sim(x, y):
    if x == y:
        return 1
    elif x is None or y is None:
        return NONE_MATCH
    else:
        return 0


class PD(IteratedExperiment):

    def setup(self):
        move_attrs = (["opp-" + str(i) for i in range(1, LAG + 1)] +
                      ["own-" + str(i) for i in range(1, LAG + 1)])
        self.agent = Agent(["move"] + move_attrs,
                           mismatch_penalty=MISMATCH,
                           noise=NOISE,
                           decay=DECAY)
        self.agent.dimilarity(move_attrs, move_sim)
        self.init_prevs()
        self.agent.populate(self.choices(), PREPOPULATED)

    def init_prevs(self):
        self.opp_prev = [None] * LAG
        self.own_prev = [None] * LAG

    def choices(self):
        return [[move] + lst
                for move, lst in zip(MOVES, repeat(self.opp_prev + self.own_prev))]


    def run_participant_prepare(self, participant, condition, context):
        self.agent.reset(True)

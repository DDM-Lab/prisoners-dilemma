# Copyright 2024 Carnegie Mellon University

from itertools import repeat
import matplotlib.pyplot as plt
import pandas as pd
from pyibl import Agent
from tqdm import trange

PARTICIPANTS = 50
ROUNDS = 100
LAG = 3
MISMATCH = 1
NOISE = 0.25
DECAY = 0.5
NONE_MATCH = 0.5
PREPOPULATED = 12

MOVES = ("D", "C")

T = 10
R = 1
P = -1
S = -10
assert(T > R > P > S)
PAYOFFS = (((P, P), (T, S)),
           ((S, T),  (R, R)))

def move_sim(x, y):
    if x == y:
        return 1
    elif x is None or y is None:
        return NONE_MATCH
    else:
        return 0

def shift(element, list):
    # Adds element to the front of the list, shifting the existing elements towards the
    # back, with the oldest element falling off the end of the list.
    if LAG:
        list.pop()
        list.insert(0, element)


class Player:

    def __init__(self):
        move_attrs = (["opp-mv" + str(i) for i in range(1, LAG + 1)] +
                      ["own-mv" + str(i) for i in range(1, LAG + 1)] +
                      ["opp-pay" + str(i) for i in range(1, LAG + 1)] +
                      ["own-pay" + str(i) for i in range(1, LAG + 1)])
        self._agent = Agent(["move"] + move_attrs,
                            mismatch_penalty=MISMATCH,
                            noise=NOISE,
                            decay=DECAY)
        self._agent.similarity(move_attrs, move_sim)
        self.reset()
        self._agent.populate(self.choices(), PREPOPULATED)

    def reset(self):
        self._agent.reset(True)
        self._opp_prev_mv = [None] * LAG
        self._own_prev_mv = [None] * LAG
        self._opp_prev_pay = [None] * LAG
        self._own_prev_pay = [None] * LAG

    def choices(self):
        return [[move] + lst
                for move, lst in zip(MOVES, repeat(self._opp_prev_mv + self._own_prev_mv +
                                                   self._opp_prev_pay + self._own_prev_pay))]

    def choose(self):
        result = self._agent.choose(self.choices())[0]
        shift(result, self._own_prev_mv)
        return result

    def respond(self, opp_move, own_payoff, opp_payoff):
        shift(opp_move, self._opp_prev_mv)
        shift(opp_move, self._opp_prev_pay)
        shift(opp_move, self._own_prev_pay)
        self._agent.respond(own_payoff)


def main():
    results = []
    players = [Player(), Player()]
    for part in trange(1, PARTICIPANTS + 1):
        for p in players:
            p.reset()
        for round in range(1, ROUNDS + 1):
            choices = [p.choose() for p in players]
            payoffs = PAYOFFS[MOVES.index(choices[0])][MOVES.index(choices[1])]
            for p, own_pay, opp_pay, opp_mv in zip(players, payoffs, reversed(payoffs), reversed(choices)):
                p.respond(opp_mv, own_pay, opp_pay)
            results.append((part, round, choices[0], choices[1], payoffs[0], payoffs[1]))
    df = pd.DataFrame(results,
                      columns=("participant pair,round,"
                               "player 1 choice,player 2 choice,"
                               "player 1 outcome,player 2 outcome").split(","))
    df["cooperation"] = ((df["player 1 choice"] == "C").astype(int) +
                         (df["player 2 choice"] == "C").astype(int))
    (df.groupby("round")["cooperation"].mean() / 2).plot(figsize=(10, 6),
                                                         ylim=(-0.03, 1.03),
                                                         xlabel="round",
                                                         ylabel="fraction of individuals picking cooperation",
                                                         title=("Prisoner's Dilemma individuaol coöperation by round\n"
                                                                f"{PARTICIPANTS} participant pairs, T={T}, R={R}, P={P}, S={S}\n"
                                                                f"lag={LAG}, noise={NOISE}, decay={DECAY}, mismatch={MISMATCH}"))
    plt.show()


if __name__ == '__main__':
    main()

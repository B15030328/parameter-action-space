from typing import List

class simpAgent:
    def __init__(self, index: int, action : float, ttype : float, isMatch : int):
        self.index = index
        self.action = action
        self.type = ttype
        self.isMatch = isMatch
        self.reward = 0
        self.isTran = 0

    def __lt__(self, other):
        if self.action < other.action:
            return True
        else:
            return False


class OriginalAgent:
    def __init__(self, index : int, action):
        self.index = index
        self.action = action


if __name__ == '__main__':
    s = []
    s.append(simpAgent(0, 0.9, 4, 1))
    s.append(simpAgent(1, 1.5, 4, 1))
    s.append(simpAgent(2, 0.4, 4, 1))
    s.append(simpAgent(2, 0.3, 4, 1))
    s.append(simpAgent(2,2.2, 4, 1))
    s.append(simpAgent(2, 3.6, 4, 1))
    s = sorted(s)
    for i in range(len(s)):
        print(s[i].action)
        print(s[i].index)
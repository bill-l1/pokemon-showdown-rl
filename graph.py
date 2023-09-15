import json
from config import BATTLE_FORMAT
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open(f'./results/dqn_{BATTLE_FORMAT}_log.json')
    data = json.load(f)
    plt.plot(data['mean_q'])
    plt.xlabel('# of battles')
    plt.ylabel('Mean Q-value')
    plt.savefig(f'./results/dqn_{BATTLE_FORMAT}_mean_q')

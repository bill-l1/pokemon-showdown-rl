# Deep Reinforcement Learning for Pokémon Battles

![Pokémon battling.](banner.jpg)

Repository for training a Deep-Q-Network (DQN) to learn how to play Pokémon battles. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Training](#training)
- [Future Work](#future-work)
- [License](#license)

## Introduction

In this game, two players battle against each other with a team of up to 6 Pokémon. Battles are turn-based, with each side selecting their moves simultaneously. The goal is to get each opposing Pokémon to faint, which is done by reducing their HP (health points) to 0.

This repo uses [Pokémon Showdown](https://pokemonshowdown.com/) for battle simulation. These agents are trained to play "random" battle formats, which gives each player a team of 6 Pokémon, semi-randomly picked to ensure a relatively fair and balanced experience. 

Although the agents can be trained to play most formats (Gens 4-9), efforts are focused on optimizing Generation 9 (Pokémon Scarlet and Violet) random battles, the most played format.

## Installation

1. Create a virtual environment:
```bash
python3 -m venv .venv
```

2. Activate the virtual environment:
- macOS and Linux:
```bash
source .venv/bin/activate
```
- On Windows (PowerShell):

```bash
.\venv\Scripts\Activate
```

3. Install project dependencies
```bash
pip install -r requirements.txt
```

4. Clone and install a local version of Pokémon Showdown

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
```

## Training

1. Start the local Pokémon Showdown server

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

2. (Optional) Configure hyperparameters in `/config.py`.

3. Run the training script

```bash
.venv/bin/python ddqn.py
```

## Future Work

- Refactoring
- Improve Pokémon embedding
- Improve training speed

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

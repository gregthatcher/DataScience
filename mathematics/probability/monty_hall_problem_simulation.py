'''
A simulation of the Monty Hall Problem
You are given three doors to choose from, and one of them has a prize.
You choose one of the doors, but you haven't yet seen if it's the prize.
You are shown another door which didn't have the prize
Should you switch your choice to the other door, or stay
with your current choice
Answer: You should switch doors
'''

import random


n_simulations = 1000000
n_doors = 3

number_of_stay_wins = 0
number_of_switch_wins = 0

for counter in range(n_simulations):
    # Let's start with no doors having prizes
    door_state = [False, False, False]
    # Pick one door to have a prize
    winning_door = random.randint(0, n_doors-1)
    door_state[winning_door] = True
    # Which door did the contestant choose first
    user_choice = random.randint(0, n_doors-1)
    # Find a "blank" door to show the contestant
    while True:
        blank_door = random.randint(0, n_doors-1)
        if (blank_door != user_choice and door_state[blank_door] is False):
            break
    other_door = 6 - (user_choice+1) - (blank_door + 1)
    other_door -= 1

    if user_choice == winning_door:
        number_of_stay_wins += 1
    elif other_door == winning_door:
        number_of_switch_wins += 1
    else:
        raise AssertionError("This should never happen!")

print()
print(f"After running {n_simulations} simulations...")
print(f"Percentage of \"Stay Wins\": {number_of_stay_wins/n_simulations:0.2}")
print(
    f"Percentage of \"Switch Wins\": {number_of_switch_wins/n_simulations:0.2}"
)
print()

import os
import pickle
import random
import re
import tqdm

import numpy as np


hunger_levels = {
    "Satiated": 0,
    "": 1,
    "Hungry": 2,
    "Weak": 3,
    "Fainting": 4,
    "Fainted": 5,
    "Starved": 6,
}
hunger_to_str = {
    0: "Satiated",
    1: "Normal",
    2: "Hungry",
    3: "Weak",
    4: "Fainting",
    5: "Fainted",
    6: "Starved",
}
labels = ["Dungeon Level", "Gold", "HP", "XP", "Hunger"]


def get_llm_str(msgs, stats):
    stat1, stat2 = stats[0], stats[1]

    diffstats = []   
    for i, label in enumerate(labels):
        if label == "HP":
            # Combine HP and Max HP
            hp1 = f"{stat1[2]}/{stat1[3]}"
            hp2 = f"{stat2[2]}/{stat2[3]}"
            change = "did not change"
            if hp1 != hp2:
                if stat1[2] < stat2[2]:
                    change = "went up from"
                elif stat1[2] > stat2[2]:
                    change = "went down from"
                diffstats.append(f"{label}: {change} {hp1} to {hp2}")
            else:
                diffstats.append(f"{label}: {change}")
        elif label == "XP":
            # Adjust index to 4 because HP and Max HP are combined
            if stat1[4] == stat2[4]:
                diffstats.append(f"{label}: did not change")
            else:
                if stat1[4] < stat2[4]:
                    diffstats.append(f"{label}: went up from {stat1[4]} to {stat2[4]}")
                else:
                    diffstats.append(f"{label}: went down from {stat1[4]} to {stat2[4]}")
        elif label == "Hunger":
            # Adjust index to 5 because of combined HP and Max HP
            hunger1 = hunger_to_str.get(stat1[5], "Unknown")
            hunger2 = hunger_to_str.get(stat2[5], "Unknown")
            if hunger1 == hunger2:
                diffstats.append(f"{label}: did not change.")
            else:
                diffstats.append(f"{label}: changed from {hunger1} to {hunger2}")
        else:
            # For Dungeon Level and Gold, indices are not affected by the HP and Max HP combination
            if stat1[i] == stat2[i]:
                diffstats.append(f"{label}: did not change")
            else:
                if label == "Dungeon Level":
                    # Dungeon levels increase as they numerically decrease
                    direction = "up" if stat1[i] > stat2[i] else "down"
                else:
                    direction = "up" if stat1[i] < stat2[i] else "down"
                diffstats.append(f"{label}: went {direction} from {stat1[i]} to {stat2[i]}")
    
    diffmsg = []
    diffmsg.append(f"Message: previously seen \"{msgs[0]}\", currently displayed \"{msgs[1]}\"")

    return diffmsg + diffstats

def convert_to_ascii_padded_numpy(input_string):
    # Convert string to a numpy array of ASCII values
    ascii_array = np.array([ord(char) for char in input_string], dtype=np.uint16)

    # Create a zero-padded array of fixed length 256
    padded_ascii_array = np.zeros(256, dtype=np.uint16)
    padded_ascii_array[:len(ascii_array)] = ascii_array

    return padded_ascii_array

def get_arrays(msgs, stats):
    msgs_arr = np.array(list(map(convert_to_ascii_padded_numpy, msgs)))
    stats_arr = np.delete(np.array(stats, dtype=np.float32), 3, axis=1) # remove max health
    diffstats = np.clip(stats_arr[1] - stats_arr[0], -1, 1)
    diffstats[1:] = 0 # mask some diffstats when training the bradley-terry model
    return {'messages': msgs_arr, 'stats': stats_arr, 'diffstats' : diffstats, 'stats_list': stats}

def get_dataset(filename):
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)

    dataset = []
    for sample in tqdm.tqdm(saved_data):
        msgs = sample['msg_strs']
        stats = sample['stats_list']
        llm_strs = get_llm_str(msgs, stats)
        arrs = get_arrays(msgs, stats)
        dataset.append({'llm_strs': llm_strs} | arrs | {'msg_strs': msgs})
    pairs_size = int(len(dataset) / 2)
    return dataset, pairs_size

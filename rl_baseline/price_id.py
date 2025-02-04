def is_item_identified(message):
    # Dictionary of items with their corresponding values
    item_prices = {
        "potion": {
            50: ["Juice", "Sickness", "Booze", "See Invis"],
            100: ["Restoration of Abilities", "Confusion", "Hallucination", "Healing", "Extra Healing", "Sleep"],
            150: ["Blindness", "Invisibility", "Monster Detection", "Object Detection", "Gain Energy"],
            200: ["Speed", "Levitation", "Enlightenment", "Full Healing", "Polymorph"],
            250: ["Acid", "Oil"],
            300: ["Gain Ability", "Paralysis", "Gain Level"]
        },
        "scroll": {
            20: ["Identify"],
            50: ["Light"],
            60: ["Enchant Weapon"],
            80: ["Remove Curse", "Enchant Armor"],
            100: ["Destroy Armor", "Confuse Monster", "Scare Monster", "Teleportation", "Gold Detection", "Food Detection", "Magic Mapping", "Fire"],
            200: ["Create Monster", "Taming", "Amnesia", "Earth"],
            300: ["Genocide", "Punishment", "Charging", "Stinking Cloud"]
        },
        "ring": {
            100: ["Adornment", "Protection", "Stealth", "Sustain Ability", "Hunger", "Warning", "Protection from Shape Changer"],
            150: ["Increase Strength", "Increase Constitution", "Increase Accuracy", "Increase Damage", "Aggression", "Poison Resistance", "Cold Resistance", "Shock Resistance", "Invisibility", "See Invisibility"],
            200: ["Regeneration", "Searching", "Levitation", "Fire Resistance", "Free Action", "Slow Digestion", "Teleportation"],
            300: ["Conflict", "Teleport Control", "Polymorph", "Polymorph Control"]
        },
        "cloak": {
            50: ["Protection", "Displacement"],
            60: ["Magic Resistance", "Invisibility"]
        },
        "boots": {
            8: ["Elven Boots", "Kicking Boots"],
            30: ["Levitation", "Fumbling"],
            50: ["Jumping", "Speed", "Water Walking"]
        },
        "wand": {
            100: ["light, nothing"],
            150: ["Invisibility"],
            175: ["Sleep"],
            200: ["Cancellation", "Teleportation"],
            500: ["Death"],
        },
        "helmet": {
            10: ["Non-magical"],
            50: ["Brilliance", "Opposite Alignment", "Telepathy"]
        }
    }

    # Split the message into words
    words = message.split()

    # Find the word "gold" to get the price and item type position
    for_index = words.index("gold")

    try:
        price = int(words[for_index - 1])
    except ValueError:
        return False  # If the price is not an integer, return False

    if 'scroll' in message:
        sell_index = words.index("your")
        item_type = words[sell_index+1]
    else:
        sell_index = words.index("Sell")
        item_type = words[sell_index-1].replace('.', '')

    if item_type not in item_prices:
        return False

    if item_type is None:
        return False

    if price * 2 in item_prices[item_type]:
        return True  # Identified by price

    return False
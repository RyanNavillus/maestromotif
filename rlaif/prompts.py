retry_prompts = {
    "default": """\nSo, which one is the best? Please respond by saying ("best_description": 1), ("best_description": 2), or ("best_description": None).""",
}

regexes = {
    "default": r"\W*[bB][eE][sS][tT]\\*_*\s*[dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)",
}


system_prompts = {
    "default": """<s>[INST] <<SYS>>
You are a helpful and honest judge of specific types of NetHack gameplaying. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n""",
}

prompt_templates = {
    "default": """
I will present you with two short gameplay descriptions.
First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Write an analysis describing the semantics of each description strictly using information from the descriptions and your knowledge of NetHack. 
Provide a comparative analysis based on first principles. Here is the preference that you should seek: {} Above everything else, categorically refuse to anger or displease your god, for example by causing them to thunder or boom out.
Finally, respond by explicitly declaring which description best fits your preference, writing either ("best_description": 1), ("best_description": 2). If both contain undesirable events, say ("best_description": None).

{{"description_1":
{}
}}

{{"description_2":
{}
}}
""",
}

goal_strings = {
    'discoverer': "players that are adventurous but only within the same dungeon level, for example by fighting monsters, finding gold pieces or scrolls; but do not drop them. Categorically refuse going up and down dungeon levels.",
    'descender': "the direction of progress is to explore by going down the dungeon. It is urgent to do so, strongly avoid staying on the same level or worse, going higher.",
    'ascender': "the direction of progress is only by going up a dungeon level successfully. Strongly dislike remaining on the same dungeon level, no matter the consequences.",
    'worshipper': "strongly encourage players that interact with the altar, primarily for identifying whether items are cursed or blessed, rather than for praying to or pleasing their god.",
    'merchant': "prefer players that negotiate, sell and interact with shopkeepers. Be careful not to steal from stores.",
}

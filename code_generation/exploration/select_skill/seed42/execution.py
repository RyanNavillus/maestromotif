
class NetHackPlayer:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.explored_levels = set()
        self.direction = 'down'  # Start by going down

    def merchant_precondition(self):
        # Placeholder for actual merchant precondition logic
        return False

    def worshipper_precondition(self):
        # Placeholder for actual worshipper precondition logic
        return False

    def select_skill(self, current_skill, dungeon_depth, merchant_precondition, worshipper_precondition):
        if merchant_precondition:
            return 'merchant'
        if worshipper_precondition:
            return 'worshipper'

        if current_skill == 'discoverer':
            self.explored_levels.add(dungeon_depth)
            if self.direction == 'down':
                if dungeon_depth < self.max_depth:
                    return 'descender'
                else:
                    self.direction = 'up'
                    return 'ascender'
            elif self.direction == 'up':
                if dungeon_depth > 1:
                    return 'ascender'
                else:
                    self.direction = 'down'
                    return 'descender'
        elif current_skill == 'descender':
            return 'discoverer'
        elif current_skill == 'ascender':
            return 'discoverer'
        else:
            return 'discoverer'


# Starting conditions
skill = 'discoverer'
dungeon_depth = 1

max_depth = 4 # to be defined

player = NetHackPlayer(max_depth)


for turn in range(15):

    print(f"Turn {turn + 1}: Skill = {skill}, Dungeon depth = {dungeon_depth}")

    merchant_precondition = player.merchant_precondition()
    worshipper_precondition = player.worshipper_precondition()

    skill = player.select_skill(skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    # the environment updates the dungeon_depth
    if skill == 'descender':
        dungeon_depth += 1
    elif skill == 'ascender':
        dungeon_depth -= 1

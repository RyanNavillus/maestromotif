import random

class NetHackPlayer:
    def __init__(self, max_depth, branch_depth):
        self.max_depth = max_depth
        self.branch_depth = branch_depth
        self.explored_levels = set()
        self.direction = 'down'  # Start by going down
        self.gold_pieces = 0
        self.monsters_defeated = 0

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

    def reach_dungeons_of_doom(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        if dungeon_depth == self.branch_depth:
            if branch_number == 2:
                return 'ascender'
            else:
                return 'descender'
        elif branch_number == 2 and dungeon_depth == self.branch_depth + 1:
            return 'ascender'
        else:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    def reach_gnomish_mines(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        if branch_number == 0:
            if dungeon_depth == self.branch_depth:
                return 'descender'
            elif dungeon_depth == self.branch_depth + 1:
                return 'ascender'
        elif branch_number == 2:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)
        return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    def perform_task(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        if self.gold_pieces >= 20 and self.monsters_defeated >= 25:
            return "ascender"

        if branch_number == 0:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)
        elif branch_number == 2:
            return self.reach_dungeons_of_doom(current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition)

    def update_gold(self, gold):
        self.gold_pieces = gold

    def defeat_monster(self):
        self.monsters_defeated += 1

# Starting conditions
skill = 'discoverer'
dungeon_depth = 1
branch_number = 0

max_depth = 3

# the environment decides on the branch_depth
branch_depth = 2

player = NetHackPlayer(max_depth, branch_depth)

for turn in range(20):
    print(f"Turn {turn + 1}: Skill = {skill}, Dungeon depth = {dungeon_depth}, Branch Number = {branch_number}, Gold pieces = {player.gold_pieces}, Monsters defeated = {player.monsters_defeated}")

    if player.gold_pieces >= 20 and player.monsters_defeated >= 25 and dungeon_depth == 0:
        print("Task is completed!")
        break

    merchant_precondition = player.merchant_precondition()
    worshipper_precondition = player.worshipper_precondition()

    skill = player.perform_task(skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition)

    # the environment updates the dungeon_depth
    if skill == 'descender':
        dungeon_depth += 1
    elif skill == 'ascender':
        dungeon_depth -= 1

    # the environment modifies the branch_number when descending from branch_depth
    if skill == "descender" and dungeon_depth == branch_depth+1:
        branch_number = random.choice([0, 2])
    # If player ascends from gnomish mines, it necessarily goes back to the dungeons of doom
    elif skill == "ascender" and dungeon_depth == branch_depth and branch_number == 2:
        branch_number = 0

    # additional method calls to update attributes
    gold = random.randint(1, 5)
    player.update_gold(gold)
    player.defeat_monster()

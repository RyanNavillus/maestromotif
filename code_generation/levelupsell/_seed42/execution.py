import random

class NetHackPlayer:
    def __init__(self, max_depth, branch_depth):
        self.max_depth = max_depth
        self.branch_depth = branch_depth
        self.explored_levels = set()
        self.direction = 'down'  # Start by going down
        self.xp_level = 1  # Start at XP level 1
        self.items_sold = False  # Track if items have been sold

    def merchant_precondition(self, dungeon_depth):
        # Shopkeepers only appear after dungeon depth 1
        return dungeon_depth > 1

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
        if self.xp_level < 4:
            return 'discoverer'
        elif not self.items_sold:
            if merchant_precondition:
                self.items_sold = True
                return 'merchant'
            else:
                return self.reach_dungeons_of_doom(current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition)
        else:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    def update_xp_level(self, xp_level):
        self.xp_level = xp_level

# Starting conditions
skill = 'discoverer'
dungeon_depth = 1
branch_number = 0
xlvl = 1

max_depth = 10 # to be defined

# the environment decides on the branch_depth
branch_depth = 2

player = NetHackPlayer(max_depth, branch_depth)

for turn in range(20):
    print(f"Turn {turn + 1}: Skill = {skill}, Dungeon depth = {dungeon_depth}, Branch Number = {branch_number}, XP Level = {player.xp_level}")

    if player.xp_level >= 4 and player.items_sold:
        print("Task was completed!")
        break

    merchant_precondition = player.merchant_precondition(dungeon_depth)
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
    if skill == 'discoverer':
        xlvl += 1
        player.update_xp_level(xlvl)
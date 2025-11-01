from game_object import GameObject

class Scene:
    def __init__(self):
        self.root = GameObject("Root")
        self.game_objects = []

    def add_game_object(self, game_object):
        self.game_objects.append(game_object)
        game_object.parent = self.root
        self.root.children.append(game_object)

    def remove_game_object(self, game_object):
        self.game_objects.remove(game_object)
        self.root.children.remove(game_object)

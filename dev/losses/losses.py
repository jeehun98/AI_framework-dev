class CategoricalCrossentropy():
    def __init__(self, name="categorical_crossentropy"):
        self.name = name

    def get_config(self):
        return {
            "name": self.name,
        }
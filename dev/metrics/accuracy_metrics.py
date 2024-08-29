class Accuracy():
    def __init__(self, name="accuracy"):
        self.name = name

    def get_config(self):
        return {"name" : self.name}
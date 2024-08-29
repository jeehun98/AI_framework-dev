class Parent:
    def __new__(cls, *args, **kwargs):
        print("Parent __new__ called")
        instance = super().__new__(cls)
        return instance

class Child(Parent):
    def __new__(cls, *args, **kwargs):
        print("Child __new__ called")
        instance = super().__new__(cls)
        return instance

child = Child()

"""
Child __new__ called
Parent __new__ called
"""
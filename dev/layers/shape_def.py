# dev/backend/shape_def.py (새로 생성)

class Shape:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        return f"Shape({self.rows}, {self.cols})"

class HistoryStack:
    def __init__(self, max_size=20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_size = max_size

    def push(self, state):
        if len(self.undo_stack) >= self.max_size:
            self.undo_stack.pop(0)  # Remove the oldest state
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo stack on new action

    def undo(self):
        if self.can_undo():
            state = self.undo_stack.pop()
            self.redo_stack.append(state)
            return state
        return None

    def redo(self):
        if self.can_redo():
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            return state
        return None

    def can_undo(self):
        return len(self.undo_stack) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0
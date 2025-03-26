class Plugin:
    def __init__(self, name):
        self.name = name
        self.methods = ["example_method"]

    async def example_method(self, message, state):
        await message.answer(f"Метод {self.name}.example_method вызван!")

def load_plugins():
    return [Plugin("TestPlugin")]
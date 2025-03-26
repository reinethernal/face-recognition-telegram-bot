from telegram.ext import CommandHandler

class Plugin:
    def __init__(self):
        self.name = "Example Plugin"
        self.methods = ["example"]
    
    def example(self, update, context):
        update.message.reply_text('Это пример команды из плагина.')
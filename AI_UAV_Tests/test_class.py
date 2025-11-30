class TestClass:
   def __init__(self, my_message, count=5):
      self.message = my_message
      self.count = count

   def sendMsg(self):
      print(self.count)
import pubsub
from pubsub import PubSub

communicator = PubSub()
messageQueue = communicator.subscribe('test')
communicator.publish('test', 'Hello World!')

o = messageQueue.listen()
o2 = next(o)
print(o2)
print(o2['data'])

print(next(messageQueue.listen()))['data']
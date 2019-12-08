from abc import ABC, abstractmethod


class Engine:
    pass


class ObservableEngine(Engine):
    def __init__(self):
        self.subscribers = set()

    def subscribe(self, subscriber):
        self.subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        self.subscribers.remove(subscriber)

    def notify(self, message):
        for subscriber in self.subscribers:
            subscriber.update((message['title'], message['text']))


class AbstractObserver(ABC):

    @abstractmethod
    def update(self):
        pass


class ShortNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = set()

    def update(self, achievement):
        self.achievements.add(achievement[0])


class FullNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = []

    def update(self, achievement):
        achievement = {'title': achievement[0], 'text': achievement[1]}
        if not achievement in self.achievements:
            self.achievements.append(achievement)


# test

# shortprinter = ShortNotificationPrinter()
# fullprinter = FullNotificationPrinter()
# engine = ObservableEngine()
# engine.subscribe(shortprinter)
# engine.subscribe(fullprinter)
# engine.notify({"title": "Воин", "text": "Дается при убийстве всех противников"})
# shortprinter.achievements, fullprinter.achievements
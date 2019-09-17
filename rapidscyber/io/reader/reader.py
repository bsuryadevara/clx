from abc import ABC, abstractmethod

class Reader(ABC):
    @property
    def has_data(self):
        return self._has_data

    @has_data.setter
    def has_data(self, val):
        self._has_data = val

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def fetch_data(self):
        pass
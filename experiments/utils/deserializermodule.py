import pickle


class DeserializerModule:

    def deserialize(self, fileName):
        loaded_data = {}
        with open(fileName, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

import pickle
import datetime

class SerializerModule:

    COLLECTOR_B = 'collectorb'
    COLLECTOR_S = 'collectors'
    COLLECTOR_F = 'collectorf'
    COLLECTOR_REWARD = 'collectorreward'
    COLLECTOR_BQ = 'collectorbq'
    COLLECTOR_SQ = 'collectorsq'
    COLLECTOR_FQ = 'collectorfq'
    COLLECTOR_BQ_BASELINE = 'collectorbq_baseline'
    COLLECTOR_SQ_BASELINE = 'collectorsq_baseline'
    COLLECTOR_FQ_BASELINE = 'collectorfq_baseline'
    COLLECTOR_LOSS = 'collectorloss'
    PENULTIMATE_FEATURES = 'penultimate_features'

    def __init__(self, directory, run_id):
        self.fileName = directory + str(datetime.datetime.now()) + "-" + str(run_id) + ".txt"
        self.serializerMap = {}

    def add_to_serializer(self, name, value):
        self.serializerMap[name] = value

    def serialize(self):
        with open(self.fileName, 'wb') as file:
            pickle.dump(self.serializerMap, file)




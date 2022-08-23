from dataclasses import dataclass
import json

from src.package import *
from src.query import *
from src.payload import *
from src.http import *


class configParser():

    def __init__(self, config_path):
        with open(config_path) as json_file:
            self.config = json.load(json_file)

        self.__packages_config = self.config["imports"]
        self.__query_config = self.config["query"]
        self.__series_config = self.config["series"]
        self.__payload_config = self.config["payload"]
        self.__http_config = self.config["http"]

    def buildTaskObject(self):
        
        imports =  Packages(self.__packages_config)
        query = self.__buildQueryObject()
        series = [ Serie(**serie) for serie in self.__series_config ]
        payload = JsonPayload(**self.__payload_config)
        http =  ApiCall(**self.__http_config)

        return {"imports" : imports, "query" : query, "series": series, "payload": payload, "http" : http}

    def __buildQueryObject(self):    
        
        name = self.__query_config["name"]
        select = Select(self.__query_config["select"]["bucket"])
        time_range = Range(Time(**self.__query_config["range"]["start"]), 
                           Time(**self.__query_config["range"]["stop"]))

        filter = Filter([Field(**field_params) for field_params in self.__query_config["filter"]["fields"]])

        return influxQuery(name, select, time_range, filter)

    
    
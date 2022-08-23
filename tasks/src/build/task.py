from dataclasses import dataclass
import json

from src.build.package import *
from src.build.query import *
from src.build.payload import *
from src.build.http import *


@dataclass
class Task:

    description: str
    imports: Packages
    options: Options
    query: influxQuery
    series: List[Serie]
    payload: JsonPayload
    http: ApiCall

    def generateTask(self) -> str:
        response = f""
        for item in self.__dataclass_fields__:
            attr = getattr(self,item)
            if isinstance(attr, list):
                for list_item in attr:
                    response+=f"{list_item.toInfluxFormat()}"
            elif not isinstance(attr, str):
                response+= f"{attr.toInfluxFormat()}"
        
        return response


class taskFactory():

    def __init__(self):
        pass
    def buildTaskObject(self, config_path) -> Task:

        with open(config_path) as json_file:
            config = json.load(json_file)

        self.__option_config = config["options"]
        self.__packages_config = config["imports"]
        self.__query_config = config["query"]
        self.__series_config = config["series"]
        self.__payload_config = config["payload"]
        self.__http_config = config["http"]

        description = config["description"]
        imports =  Packages(self.__packages_config)
        options = Options(self.__option_config["name"], Time(**self.__option_config["every"]),Time(**self.__option_config["offset"]))
        query = self.__buildQueryObject()
        series = [ Serie(**serie) for serie in self.__series_config ]
        payload = JsonPayload(**self.__payload_config)
        http =  ApiCall(**self.__http_config)

        return Task(description,imports,options,query,series,payload,http)

    def __buildQueryObject(self) -> influxQuery:    
        
        name = self.__query_config["name"]
        select = Select(self.__query_config["select"]["bucket"])
        time_range = Range(Time(**self.__query_config["range"]["start"]), 
                           Time(**self.__query_config["range"]["stop"]))

        filter = Filter([Field(**field_params) for field_params in self.__query_config["filter"]["fields"]])

        return influxQuery(name, select, time_range, filter)

        
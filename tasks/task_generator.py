from dataclasses import dataclass, fields
from typing import List

from src.package import Packages
from src.query import influxQuery
from src.files import configParser
from src.payload import Serie, JsonPayload
from src.http import ApiCall

@dataclass
class Task:

    imports: Packages
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
            else:
                response+= f"{attr.toInfluxFormat()}"
        
        return response




if __name__ == "__main__":

    config_path = "./config_example.json"
    parser = configParser(config_path)
    task = Task(**parser.buildTaskObject())
    
    print(task.generateTask())



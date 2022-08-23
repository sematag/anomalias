from dataclasses import dataclass
from typing import List



@dataclass
class Serie():

    name: str
    query_name: str
    column_name: str

    def toInfluxFormat(self) -> str:
        
        response = f""
        response+=f"{self.name} = \n\t"
        response+=f"{self.query_name}\n\t  "
        response+=f"|> findColumn(fn: (key) => true, column: {self.column_name})\n"
        
        return response 


@dataclass
class JsonPayload():

    name: str
    columns_names : List[str]

    def toInfluxFormat(self) -> str:
        
        json_dict_str = " ".join([ f"{column_name}: {column_name}," for column_name in self.columns_names]) 
        return f"\nJsonData = " + "{" + f"{json_dict_str}" + f"name: {self.name}" + "}\n"
       



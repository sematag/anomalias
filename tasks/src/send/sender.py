from dataclasses import dataclass
import json
import requests
from src.build.task import Task

@dataclass
class Sender():

    token : str
    influx_url : str
    org_id : str

    def send(self, task : Task) -> str:
        
        headers = { "Authorization" : f"Token {self.token}", "Content-Type": "application/json"}

        body = {
            "flux" : f"{task.generateTask()}",
            "orgID" : f"{self.org_id}",
            "status" : "active",
            "description" : f"{task.description}"
        }

        return requests.post(self.influx_url, headers=headers, json=body)


class senderFactory():
    
    def __init__(self):
        pass

    def buildSenderObject(self, config_path) -> Sender:
        
        with open(config_path) as json_file:
            config = json.load(json_file)

        return Sender(**config)

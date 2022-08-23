from dataclasses import dataclass
from typing import List

from src.build.query import Time

@dataclass
class Options:

    task_name: str
    every: Time
    offset: Time

    def toInfluxFormat(self) -> str:
        return "option task = "+ "{name: " + f"\"{self.task_name}\", every: {self.every.toInfluxFormat()}, offset: {self.offset.toInfluxFormat()}" + "}\n\n"

@dataclass
class Packages:

    packages: List[str]

    def toInfluxFormat(self) -> str:
        response = f""
        for package in self.packages:
            response+= f"import \"{package}\"\n"
        
        return response + "\n"
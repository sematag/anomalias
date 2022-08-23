from dataclasses import dataclass
from typing import List

@dataclass
class Packages:

    packages: List[str]

    def toInfluxFormat(self) -> str:
        response = f""
        for package in self.packages:
            response+= f"import {package}\n"
        
        return response + "\n"
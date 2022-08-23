from dataclasses import dataclass
from typing import List




@dataclass
class Select:

    bucket: str

    def toInfluxFormat(self) -> str:
        return f"from(bucket: \"{self.bucket}\")\n"


@dataclass
class Time:

    hours: int = 0
    minutes: int = 0
    seconds: int = 0

    def toInfluxFormat(self) -> str:
        return f"{self.hours}h{self.minutes}m{self.seconds}s"

@dataclass
class Range:

    start: Time
    stop: Time

    def toInfluxFormat(self) -> str:
            return f"|> range(start: -{self.start.toInfluxFormat()}, stop: -{self.stop.toInfluxFormat()})\n"



@dataclass
class Field:

    name: str
    value: str
    equal: bool

    def toInfluxFormat(self) -> str:
        
        self.connector = "==" if self.equal else "!="

        return f"r[\"{self.name}\"] {self.connector} \"{self.value}\""

@dataclass
class Filter:

    fields: List[Field]

    def toInfluxFormat(self) -> str:
        
        response = "|> filter(fn: (r) => "
        for field in self.fields:
            response += f" {field.toInfluxFormat()} or"
        
        response = " ".join((response.split(" "))[:-1]) + ")\n"
        return response



@dataclass
class influxQuery:

    name: str
    select: Select
    range : Range
    filter : Filter

    def toInfluxFormat(self) -> str:

        response = f"{self.name} = \n  "
        response+=f"{self.select.toInfluxFormat()}\t"
        response+=f"{self.range.toInfluxFormat()}\t"
        response+=f"{self.filter.toInfluxFormat()}\n"

        return response

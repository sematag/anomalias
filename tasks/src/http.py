from dataclasses import dataclass
from typing import List

@dataclass
class ApiCall():

    api_method: str
    id: str
    data_name: str

    def toInfluxFormat(self) -> str:
        
        response = f"http.post(\n\t"
        response += f"url: \"http://api:8000/{self.api_method}/{self.id}\",\n\t"
        response += "headers: {\n\t  "
        response += "Authorization: \"API kw9xrtqg56z1htwVJUJxgGoozZykmVMP3ScKcK3s0MQGYUfNClQtXwe6HL7pWX_T5N0Q0EUVo51wVl00B4Cdjw==\",\n\t  "
        response += "\"Content-type\": \"application/json\",\n\t"
        response += "},\n\t"
        response += f"{self.data_name} : json.encode(v: jsonData),\n"
        response += f")\n\n{self.data_name}"

        return response



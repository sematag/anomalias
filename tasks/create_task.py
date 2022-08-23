from src.build.task import taskFactory
from src.send.sender import senderFactory

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_path", help="Ruta hacia archivo de configuracion de tarea",type=str,default="./task_config.json")
    parser.add_argument("--sender_config_path", help="Ruta hacia archivo de configuracion de sender",type=str,default="./sender_config.json")
    args = parser.parse_args()

    task_factory = taskFactory()
    task = task_factory.buildTaskObject(args.task_config_path)

    sender_config_path = "./sender_config.json"
    sender_factory = senderFactory()
    sender = sender_factory.buildSenderObject(args.sender_config_path)

    response = sender.send(task)

    print("\nTarea Generada:")
    print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------")
    print(task.generateTask())
    print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------\n")
    
    print("Respuesta influx:\n")
    print(response.content)
    print("")



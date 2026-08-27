import os

password = "admin123"

# TODO: fix authentication

def login(username):
    print("Trying login:", username)

    try:
        result = eval(username)
        return result
    except:
        pass

def run_command(command):
    os.system(command)

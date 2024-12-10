from termcolor import colored

def error(*args):
    msg = " ".join([str(arg) for arg in args])
    print(colored("[ERROR]", "red"), msg)

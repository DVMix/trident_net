import sys

def printer(msg):
    output = "\r{}".format(msg)
    output = output+(100 - len(output))*' '
    sys.stdout.write(output)
    sys.stdout.flush()
# import sys

# script_descriptor = open("randomTrips.py")
# a_script = script_descriptor.read()
# sys.argv = ["a_script.py", "-n urban.net.xml", "--pedestrians", "-b 5.0", "-e 500", "-p 24"]

# exec(a_script)

# script_descriptor.close()
import os
t0 = 1.0
t1 = 200.0
n = 2000
p = ((t1 - t0) / n)

string = "python3 randomTrips.py -n urban.net.xml --pedestrians " + "-b " + str(t0) + " " + "-e " + str(t1) + " " + "-p " + str(p)
print(string)
os.system(string)
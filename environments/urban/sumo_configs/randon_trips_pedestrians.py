import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # noqa
import randomTrips # noqa

base_dir = os.path.dirname(os.path.realpath(__file__))

def generate_walker_trips(start_time = 0, end_time = 0, period = 0.1):
    net = os.path.join(base_dir, 'urban.net.xml')
    output_file = os.path.join(base_dir, 'pedestrian.trips.xml')
    weights_file = os.path.join(base_dir, 'weights_outprefix')
    print(output_file)
    # generate the pedestrians for this simulation
    # randomTrips.main(randomTrips.get_options([
    #     '--net-file', net,
    #     '--output-trip-file', output_file,
    #     '--seed', '42',  # make runs reproducible
    #     '--pedestrians',
    #     '--prefix', 'ped',
    #     # prevent trips that start and end on the same edge
    #     '--min-distance', '0',
    #     '--trip-attributes', 'departPos="random" arrivalPos="random" speed="1.5" ',
    #     '--binomial', '4',
    #     '--persontrip.walkfactor', str(2.0),
    #     '-b', str(start_time),
    #     '-e', str(end_time),
    #     '-p', str(period)]))
    randomTrips.main(randomTrips.get_options([
        '--net-file', net,
        '--output-trip-file', output_file,
        '--pedestrians',
        '--trip-attributes', 'departPos="random" arrivalPos="random" speed="2.0" ',
        '--weights-prefix', weights_file,
        '-b', str(start_time),
        '-e', str(end_time),
        '-p', str(period)]),)


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    print(base_dir)

    t0 = 0.0
    t1 = 30.0
    n = 1000
    p = ((t1 - t0) / n)
    print(p)

    generate_walker_trips(start_time = t0, end_time = t1, period = p)


#     import os
# t0 = 1.0
# t1 = 10.0
# n = 50
# p = ((t1 - t0) / n)

# string = "python3 randomTrips.py -n urban.net.xml --pedestrians " + "-b " + str(t0) + " " + "-e " + str(t1) + " " + "-p " + str(p) + " " + "--trip-attributes " + "departPos="random""
# print(string)
# os.system(string)
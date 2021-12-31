#!/usr/bin/env python

import paho.mqtt.client as mqtt
import subprocess

from pixel_ring import pixel_ring
from gpiozero import LED

import pathlib2
import time

from generic_audio_functions import record_sound


# Sound recording
def recording(args):
    args = args.split()

    # parseing
    test = '-t' in args
    LoS = '--LoS' in args
    top = '--top' in args

    music_name = args[args.index('--music') + 1]
    duration = float(args[args.index('--duration') + 1])

    # make file name for recorded file
    file_name = 'rec_{:03d}cm_{:03d}'.format(int(args[args.index('--distance') + 1]),
                                             int(args[args.index('--direction') + 1]))

    if LoS:
        file_name = '{}_euclid_{:03d}cm_{:03d}'.format(file_name,
                                                       int(args[args.index('--edistance') + 1]),
                                                       int(args[args.index('--edirection') + 1]))
        LoS_state = 'Non_Line_of_Sight'
    else:
        LoS_state = 'Line_of_Sight'

    file_name = '{}_loc{}.wav'.format(file_name,
                                      args[args.index('--location') + 1])

    file_name = file_name.replace('.wav', f'{time.time()}.wav')

    # determine storage location and make it if it doesn't exist

    if top:
        top_state = 'Obstructed_Top'
    else:
        top_state = 'Free_Top'

    file_path = '../Recorded_files/{}/{}/{}/Raw_recordings'.format(top_state, LoS_state, music_name)

    pathlib2.Path(file_path).mkdir(parents=True, exist_ok=True)

    print('file name: {}'.format(file_name))
    print('file loc:  {}'.format(file_path))

    power = LED(5)
    power.on()
    pixel_ring.change_pattern('echo')
    pixel_ring.set_brightness(10)

    pixel_ring.think()
    if test:
        # test mode
        time.sleep(duration)

    else:
        record_sound(file_path + "/" + file_name, duration_s=duration, channels=8)
        # subprocess.call(['arecord', \
        #                  '-r 44100', \
        #                  '-f', 'S16_LE', \
        #                  '-c 8', \
        #                  '-d', str(int(duration)), \
        #                  '{}/{}'.format(file_path, file_name)])

    pixel_ring.off()
    power.off()


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    client.subscribe("playrec")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    msg.payload = msg.payload.decode('utf-8')
    print(msg.topic + " " + str(msg.payload))
    client.publish("recing", 0)

    recording(msg.payload)

    # used for testing
    # print "lightshow"
    # subprocess.call(['python', 'Lightshow1.py'])

    print('done')
    client.publish("rec_done", 1)


client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

# connect
client.connect("192.168.1.196")

client.loop_start()
while True:
    try:
        pass
    except KeyboardInterrupt:
        break

client.loop_stop()

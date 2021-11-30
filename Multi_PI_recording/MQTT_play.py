#!/usr/bin/env python

import paho.mqtt.client as mqtt
import subprocess
from pixel_ring import pixel_ring
from gpiozero import LED

import time

# playing music
def playing(args):
    args = args.split()

    test = '-t' in args
    
    music_name = args[args.index('--music')+1]
    music_file = '../Music_files/{}.wav'.format(music_name)
    
    duration = float(args[args.index('--duration')+1])

    
    power = LED(5)
    power.on()
    pixel_ring.set_brightness(100)
    
    pixel_ring.listen()
    
    # wait to ensure recording started
    time.sleep(1)

    pixel_ring.think()
    if test:
        #test mode
        time.sleep(duration)
    else:
        subprocess.call(['aplay', \
                         '-d', str(int(duration)),\
                         '-r 44100',\
                         music_file])
    pixel_ring.off()
    power.off()
        


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    client.subscribe("playrec")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    msg.payload = msg.payload.decode('utf-8')
    print(msg.topic+" "+str(msg.payload))

    playing(msg.payload)
    #Used for testing
    #print "lightshow"
    #subsystem.call(['python', 'Lightshow2.py'])
    print ('done')
    client.publish("play_done",1)

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

#connect
client.connect("robomindpi-002")




client.loop_start()
while True:
    try:
        pass
    except KeyboardInterrupt:
        break

client.loop_stop()

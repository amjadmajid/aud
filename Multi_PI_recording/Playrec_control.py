import paho.mqtt.client as mqtt
import time

def play_done_callback(client, userdata, message):
    global play_done
    play_done = True
    print("play_done")

def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("rec_done")

def on_message(client, userdata, message):
    print(msg.topic+" "+str(msg.payload))

def Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration):

    args = ""
    if test:
        args = "{} -t".format(args)

    args = "{} --distance {}".format(args, dist)
    args = "{} --direction {}".format(args, direction)

    if not LoS:
        args = "{} --LoS".format(args)
        args = "{} --edistance {}".format(args, edist)
        args = "{} --edirection {}".format(args, edirection)

    if top:    
        args = "{} --top".format(args)

    args = "{} --location {}".format(args, location)
    args = "{} --duration {}".format(args, duration)

    return args



# User inputs start

# (Geodesic) souce location
dist = 50 #cm
direction = 0

# Line-of-Sight state
LoS = False

# (Euclidian) source location (only send if LoS == False)
edist = 58 #cm
edirection = 16

# Meta lobation of recording
location = "Test"

# Top state (if ther's an obustructio inbetween the mics)
top = True

# Testing flag (set to True to run trough program without actually playing/recording
test = False

# Music_files
# File names
if test:
	M = 2
	chirp_types = ["0s024"]
else:
	M = 8
	chirp_types = ["0s024", "0s048"]

music_names = []
for j in range(len(num_chirp_types)):
    for i in range(M):
        music_names.append('chirp_train_chirp_{}_{}'.format(chirp_types[j],i-1))
        #music_names.append("Test{}".format(i))
    
# Length of the music files (seconds)
if test:
	duration = 2
else:
	duration = 25

# User inputs end
msg = Input_parsing(dist, direction, LoS, edist, edirection, location, top, test, duration)


rec_init = False
play_init = False

rec_done = rec_init
play_done = play_init

#connect to mqtt
client = mqtt.Client()

client.connect("Robomindpi-002")

client.subscribe("rec_done")
client.subscribe("play_done")

client.message_callback_add("rec_done", rec_done_callback)
client.message_callback_add("play_done", play_done_callback)

# loop
client.loop_start()
print("playrec settings \n{}".format(msg))
for i in range(len(music_names)):
    print(music_names[i])
    
    rec_done = rec_init
    play_done = play_init
    
    tx_args = "{} --music {}".format(msg,music_names[i])
    print(tx_args)
    client.publish("playrec", tx_args)

    while not (rec_done and play_done):
        pass
    
    time.sleep(1)

print("done")
client.loop_stop()
